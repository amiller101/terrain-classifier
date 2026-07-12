import numpy as np
from PIL import Image
import image_preprocessing
import hog 
import lbp
from operator import itemgetter
from skimage.feature import hog as ski_hog
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import os
import time
import cv2
from itertools import chain
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


GAUSSIAN_PIXEL_STD_DEV_DENOM = 0.01

def images_to_video(frame_directory, video_name, fps=15):

    images = [img for img in os.listdir(frame_directory) if img.endswith(".png")]
    # Ensure images are sorted by name
    images.sort() 

    # Read the first image to get dimensions
    frame = cv2.imread(os.path.join(frame_directory, images[0]))
    height, width, layers = frame.shape

    # Define video writer
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Add frames
    for image in images:
        video.write(cv2.imread(os.path.join(frame_directory, image)))

    cv2.destroyAllWindows()
    video.release()


def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def gaussian_vote(side_length, pixel_std_dev):
    """
    Creates a square 2D Gaussian matrix with mean of 1 at centroid.
    
    Uses pixel-based deviation to maintain similar distibution behavior for varying side_length.

    Args:
        side_length (int): Size of a patch dimension in pixels.
        pixel_std_dev (float): Variance of the gaussian distribution.

    Returns:
        np.array(side_length, side_length): vote matrix with gaussian-weighted cannonical entries
    """

    centroid = float (side_length - 1) / 2.0
    peak = gaussian_pdf(x=0, mu=0, sigma=pixel_std_dev)

    vote = np.zeros((side_length, side_length))
    for (i, j), _ in np.ndenumerate(vote):
        dist = np.sqrt((i - centroid)**2 + (j - centroid)**2)
        vote[i,j] = gaussian_pdf(x=dist, mu=0, sigma=pixel_std_dev) / peak
    
    return vote


def validate_hog(image_path):
    """Compare custom hog implementation with sci-kit hog.

    Prints shape of both produced feature vectors, as well as their L2 difference and percentage mismatch. Ignores order difference in feature vector entries.

    Args:
        image_path (string): Relative file path of an image, of which a patch is taken to evaluate.

    """

    # Select one image.
    images = image_preprocessing.unsegmented_image_patch_extraction(image_path)
    # Put image into list for my implementations, alone for sklearn's.
    image = []
    image.append(images[0])
    ski_image = images[0]
    

    # Compute both HOG
    ski_results = ski_hog(
        np.array(ski_image),
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    my_results = hog.extract_feature_vectors(image)[0]

    print("My shape:", my_results.shape)
    print("Sk-image shape:", ski_results.shape)

    # Sort both in L2 normal to account for entry ordering differences
    diff = np.linalg.norm(np.sort(my_results) - np.sort(ski_results))
    print("L2 difference:", diff)
    print("Relative difference percentage:", diff/np.linalg.norm(my_results))


def pca_reduce_train_test(
    training_features,
    testing_features,
    information_retention,
    scale_features,
):
    """PCA-style dimensionality reduction using SVD on training-only preprocessing.

    Matches usual PCA when ``scale_features`` is False: subtract training column means,
    then take the top singular directions of the centered matrix and project both sets.

    When ``scale_features`` is True, apply ``(x - mean_j) / std_j`` per column with ``mean``
    and ``std`` from **raw** training columns (same as ``StandardScaler``), then SVD on that
    matrix — suitable when mixing features with different scales (e.g. HOG + LBP).

    Variance explained uses squared singular values (energy), not raw ``S``.

    Args:
        training_features (np.ndarray): Shape ``(n_train, n_features)``.
        testing_features (np.ndarray): Shape ``(n_test, n_features)``.
        information_retention (float): Target fraction of total variance to retain in ``(0, 1]``.
        scale_features (bool): Per-feature scaling from training ``mean`` and ``std``.

    Returns:
        tuple: ``(training_reduced, testing_reduced, rank, variance_fraction_kept)``
        where outputs have shape ``(..., rank)``.
    """
    x_train = np.asarray(training_features, dtype=np.float64)
    x_test = np.asarray(testing_features, dtype=np.float64)
    mean = x_train.mean(axis=0)
    if scale_features:
        std = x_train.std(axis=0, ddof=0)
        std = np.where(std < 1e-12, 1.0, std)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    else:
        x_train = x_train - mean
        x_test = x_test - mean

    u, singular_values, vh = np.linalg.svd(x_train, full_matrices=False)
    total_energy = np.sum(singular_values ** 2)
    if total_energy <= 0:
        rank = int(vh.shape[0])
        variance_fraction_kept = 1.0
    else:
        cumulative_ratio = np.cumsum(singular_values ** 2) / total_energy
        rank = int(np.searchsorted(cumulative_ratio, information_retention, side="left") + 1)
        rank = max(1, min(rank, len(singular_values)))
        variance_fraction_kept = float(np.sum(singular_values[:rank] ** 2) / total_energy)

    training_reduced = x_train @ vh[:rank, :].T
    testing_reduced = x_test @ vh[:rank, :].T
    return training_reduced, testing_reduced, rank, variance_fraction_kept

# Transformer wrapper to allow feature extraction hyperparameters to be used in tuning.
class PatchFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, lbp_cell_size=8, hog_cell_size=8, hog_block_size=2):
        self.lbp_cell_size = lbp_cell_size
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X = list of images
        features = np.concatenate((lbp.extract_feature_vectors(X, self.lbp_cell_size), hog.extract_feature_vectors(X, self.hog_block_size, self.hog_cell_size)), axis=1)
        return np.array(features)
    

def separate_data(training, testing):
    # Flatten training only: (List(List(Image, label))) -> (List(Images, label))
    training = list(chain.from_iterable(training))
    # Seperate training images and labels
    training_images = list(map(itemgetter(0), training))
    training_labels = list(map(itemgetter(1), training))

    # Create valid mask for the testing data, preserving frame/patch organization for reconstruction.
    # In the same loop, save all valid entries to testing_valid_entries[]
    testing_valid_masks = []
    testing_valid_entries = []
    for frame in testing:
        valid_patches_mask = []
        for patch in frame:
            # Valid if non-zero label
            valid_patches_mask.append(patch[1] != 0)
            # Save if valid
            if patch[1] != 0:
                testing_valid_entries.append(patch)
        testing_valid_masks.append(valid_patches_mask)

    # Seperate valid testing images and labels
    testing_images = list(map(itemgetter(0), testing_valid_entries))
    testing_labels = list(map(itemgetter(1), testing_valid_entries))

    # Could change to dict unpacking for readability
    return training_images, training_labels, testing_images, testing_labels, testing_valid_masks 

# Note: not currently used as a baseestimator or transformer mix in
class image_preprocessor(BaseEstimator, TransformerMixin):
    # Could move many assignments from __init__ to transform()
    def __init__(self,
        frames_root="data\\RUGD_frames-with-annotations",
        ann_root="data\\RUGD_annotations",
        video=None,
        video_generalization=True,
        multi_video_training=True,
        videos=None,
        videos_train=None,
        videos_test=None,
        testing_percentage=0.30,
        patch_size=64,
        training_stride=32,
        testing_stride=32):
            self.frames_root = frames_root
            self.ann_root = ann_root
            self.video = video
            self.video_generalization = video_generalization
            self.multi_video_training = multi_video_training
            self.videos = videos
            self.videos_train = videos_train
            self.videos_test = videos_test
            self.testing_percentage = testing_percentage
            self.patch_size = patch_size
            self.training_stride = training_stride
            self.testing_stride = testing_stride


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        if self.multi_video_training:
            training, testing = image_preprocessing.segmented_directory_patch_extraction(
                self.frames_root,
                self.ann_root,
                True,
                self.video_generalization,
                self.videos,
                self.testing_percentage,
                self.patch_size,
                self.training_stride,
                self.patch_size,
                self.testing_stride,
                videos_train=self.videos_train,
                videos_test=self.videos_test,
            )
        else:
            training, testing = image_preprocessing.segmented_directory_patch_extraction(
                os.path.join(self.frames_root, self.video),
                os.path.join(self.ann_root, self.video),
                False,
                self.video_generalization,
                None,
                self.testing_percentage,
                self.patch_size,
                self.training_stride,
                self.patch_size,
                self.testing_stride,
                videos_train=self.videos_train,
                videos_test=self.videos_test,
            )
        
        return training, testing

def main():

    ############ IMAGE PREPROCESSING ####################
    multi_video_training = True
    patch_size=64
    training_stride = 16
    testing_stride = 8
    video = "trail"

    processor = image_preprocessor(
        testing_percentage=0.30,
        patch_size=patch_size,
        training_stride=training_stride,
        testing_stride=testing_stride,
        frames_root="data\\RUGD_frames-with-annotations",
        ann_root="data\\RUGD_annotations",
        video_generalization=False,
        multi_video_training=multi_video_training,
        video=video,
        videos= ["trail-3", "trail-4"],
        #videos_train= ["trail-9"],
        #videos_test = ["trail-10"]
        )
    
    training, testing = processor.transform(None)
   
    ### Experimentation

    print("Images processed")

    ########### DATA SEPERATION ##################

    # # Flatten training only: (List(List(Image, label))) -> (List(Images, label))
    # training = list(chain.from_iterable(training))
    # # Seperate training images and labels
    # training_images = list(map(itemgetter(0), training))
    # training_labels = list(map(itemgetter(1), training))

    # # Create valid mask for the testing data, preserving frame/patch organization for reconstruction.
    # # In the same loop, save all valid entries to testing_valid_entries[]
    # testing_valid_masks = []
    # testing_valid_entries = []
    # for frame in testing:
    #     valid_patches_mask = []
    #     for patch in frame:
    #         # Valid if non-zero label
    #         valid_patches_mask.append(patch[1] != 0)
    #         # Save if valid
    #         if patch[1] != 0:
    #             testing_valid_entries.append(patch)
    #     testing_valid_masks.append(valid_patches_mask)


    # # Seperate valid testing images and labels
    # testing_images = list(map(itemgetter(0), testing_valid_entries))
    # testing_labels = list(map(itemgetter(1), testing_valid_entries))
    training_images, training_labels, testing_images, testing_labels, testing_valid_masks = separate_data(training, testing)

    ## Experimentation

    print("Training and Testing Seperated")

    ############# Feature Extraction: HOG + LBP ####################
    
    # extractor = PatchFeatureExtractor()
    # training_features = extractor.transform(training_images)
    # testing_features = extractor.transform(testing_images)

    # pixel side length of image subdivision areas to form histograms from, all to be contactenated together for the full image.
    lbp_cell_size = int(np.sqrt(patch_size))
    num_neighbors = 8
    pixel_radius = 1

    train_lbp_extraction_time_start = time.perf_counter()
    training_features = lbp.extract_feature_vectors(training_images, lbp_cell_size, num_neighbors, pixel_radius)
    train_lbp_extraction_time_end = time.perf_counter()
    train_lbp_extraction_time = train_lbp_extraction_time_end - train_lbp_extraction_time_start
    print(f"Training LBP Extracted in {train_lbp_extraction_time}")

    test_lbp_extraction_time_start = time.perf_counter()
    testing_features = lbp.extract_feature_vectors(testing_images, lbp_cell_size, num_neighbors, pixel_radius)
    test_lbp_extraction_time_end = time.perf_counter()
    test_lbp_extraction_time = test_lbp_extraction_time_end - test_lbp_extraction_time_start
    print(f"Testing LBP Extracted in {test_lbp_extraction_time}")

    ### HOG: PIL.Image -> numpy.array of histogram bins.
    ### Each feature is a concatenation of block-normalized cell-local orientation histgram bins for every cell in the image.
    ### Dimension of each entry is (blocks per image) * (cells per block) * (histogram bins per cell)

    # Extract features
    hog_block_size = 2
    hog_cell_size = int(np.sqrt(patch_size))

    train_hog_extraction_time_start = time.perf_counter()
    training_features = np.concatenate((training_features, hog.extract_feature_vectors(training_images, hog_block_size, hog_cell_size)), axis=1)
    train_hog_extraction_time_end = time.perf_counter()
    train_hog_extraction_time = train_hog_extraction_time_end - train_hog_extraction_time_start
    print(f"Training HOG Extracted in {train_hog_extraction_time}")

    test_hog_extraction_time_start = time.perf_counter()
    testing_features = np.concatenate((testing_features, hog.extract_feature_vectors(testing_images, hog_block_size, hog_cell_size)), axis=1)
    test_hog_extraction_time_end = time.perf_counter()
    test_hog_extraction_time = test_hog_extraction_time_end - test_hog_extraction_time_start
    print(f"Testing HOG Extracted in {test_hog_extraction_time}")
    
    print(f"Final Feature Vector Array Shape: {training_features.shape}")


    ############# Custom PCA via SVD  ####################
    custom_PCA = False
    if (custom_PCA == True):    
        information_retention = 0.99
        # Set true when stacking features with different scales
        scale_features_before_pca = True 

        training_features, testing_features, pca_rank, pca_variance_kept = pca_reduce_train_test(
            training_features,
            testing_features,
            information_retention,
            scale_features_before_pca,
        )
        print("PCA (SVD) complete")
        print(f"rank: {pca_rank}")
        print(f"variance retained (squared singular values): {pca_variance_kept:.4f}")
    else:
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        testing_features = scaler.transform(testing_features)
        pca = PCA()
        training_features = pca.fit_transform(training_features)
        testing_features = pca.transform(testing_features)

    ### Experimentation

    ###############  BUILDING MODEL  ################
    tuning = False
    model_choice = "LogiReg"
    if (tuning == True):

        if (model_choice == "LogiReg"):
            classifier = LogisticRegression(max_iter=1000)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', classifier)
            ])
            param_grid = {
                'pca__n_components': [None],
                'clf__C': [0.01]
            }
        elif(model_choice == "RF"):
            classifier = RandomForestClassifier(n_jobs=1, class_weight='balanced')
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', classifier)
            ])
            param_grid = {
                'pca__n_components': [None],
                'clf__n_estimators': [300],
                'clf__max_depth': [None, 10, 15, 20],
                'clf__min_samples_leaf': [None, 2, 5],
                'clf__bootstrap': [True, False]
            }

        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            refit=True
        )

        model.fit(training_features, training_labels)
        print(model.best_params_)
        reconstruction_probabilities = model.predict_proba(testing_features)
        predictions = model.classes_[np.argmax(reconstruction_probabilities, axis=1)]
        print(f"\\ Accuracy: {np.mean(predictions == testing_labels)}\n")
    else:  

        # No Tuning
        if (model_choice == "LogiReg"):
            model = LogisticRegression(
                max_iter=1000,
                C=0.01
            )
            print("Training Logistic Regression")

            training_start = time.perf_counter()
            model.fit(training_features, training_labels)
            training_end = time.perf_counter()
            train_time = training_end - training_start
            print(f"Model training Complete in {train_time:.4f}")

            testing_start = time.perf_counter()
            reconstruction_probabilities = model.predict_proba(testing_features)
            predictions = model.classes_[np.argmax(reconstruction_probabilities, axis=1)]
            testing_end = time.perf_counter()
            test_time = testing_end - testing_start
            print(f"Model testing Complete in {test_time:.4f}")
            print(f"Accuracy: {np.mean(predictions == testing_labels)}\n")
        elif (model_choice == "SVM"):
            # Converts a decision function into plausible relative probablilties.
            # Probabilites are non-exact and sensitive to scale,
            # but much faster to compute than proper probabilities.
            def softmax(x):
                # translate the values towards the origin to avoid possible overflow issues with exp() on large vals
                x = x - np.max(x, axis=1, keepdims=True)
                exp_x = np.exp(x)
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)

            kernels = ['rbf']
            reconstruction_probabilities = None
            for kernel_name in kernels:
                print(f"Training on {kernel_name}")
                model = svm.SVC(kernel=kernel_name)
                print(f"{kernel_name} initialized")
                model.fit(training_features, training_labels)
                print("Training Complete")
                print(f"\\ \\ -- Model with {kernel_name} kernel --")
                print(f"\\ Num of support vectors: {len(model.support_vectors_)}")
                # Score on valid testing entries only and save predictions for reconstruction
                scores = model.decision_function(testing_features)
                # binary case → expand to 2-class scores
                if scores.ndim == 1:
                    scores = np.vstack([-scores, scores]).T
                reconstruction_probabilities = softmax(scores)
                predictions = model.classes_[np.argmax(reconstruction_probabilities, axis=1)]

                print(f"\\ Accuracy: {np.mean(predictions == testing_labels)}\n")

    #############   Segmentation Map Image Reconstruction (from testing data) #######

    # Make color lookup table. Made as dict first for convenience only.
    ID_to_color = {
        0: np.array([0, 0, 0]),       #Unknown
        1: np.array([108, 64, 20]),   #Dirt
        2: np.array([0, 102, 0]),     #Grass     
        3: np.array([64, 64, 64]),    #Asphalt
        4: np.array([255, 128, 0])    #Gravel
    }
    max_id = max(ID_to_color.keys())
    color_lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for id, color in ID_to_color.items():
        color_lut[id] = color

    # Technically a safety check...but shouldn't need? Just confirms the classes in model.classes_ map directly to our terrain ID's.
    def scatter_class_probs_to_terrain_channels(class_prob_row, model_classes, terrain_max_id):
        """Map predict_proba columns (ordered by model.classes_) onto terrain-id vote channels 0..terrain_max_id."""
        channels = np.zeros(terrain_max_id + 1, dtype=np.float32)
        for probability_column_index, terrain_class_id in enumerate(model_classes):
            terrain_id = int(terrain_class_id)
            if 0 <= terrain_id <= terrain_max_id:
                channels[terrain_id] = class_prob_row[probability_column_index]
        return channels

    terrain_maps = []
    frame_height = 550
    frame_width = 688
    pixel_standard_deviation = patch_size / GAUSSIAN_PIXEL_STD_DEV_DENOM
    vote = gaussian_vote(patch_size, pixel_standard_deviation)

    # Should parallelize below
    # For each full image frame: reconstruction[frame] is 0/1 per patch (zero vs model probabilities).
    probabilities_list_idx = 0
    for frame in testing_valid_masks:
        votes = np.zeros((frame_height, frame_width, max_id + 1), dtype=np.float32)
        patch_idx = 0
        for y in range(0, frame_height - patch_size + 1, testing_stride):
            for x in range(0, frame_width - patch_size + 1, testing_stride):

                # If patch is known and has been predicted
                if frame[patch_idx]:
                    class_prob_vec = reconstruction_probabilities[probabilities_list_idx]
                    probabilities_list_idx += 1
                
                    terrain_channel_probs = scatter_class_probs_to_terrain_channels(
                        class_prob_vec, model.classes_, max_id
                    )
                    votes[y:y+patch_size, x:x+patch_size, :] += vote[..., np.newaxis] * terrain_channel_probs

                patch_idx += 1

        # Tally votes
        label_map = np.argmax(votes, axis=2)
        terrain_map = color_lut[label_map]
        terrain_maps.append(terrain_map)

    # Convert label maps to images
    terrain_map_images = []
    for m in terrain_maps:
        m = m.astype(np.uint8)
        print("Final image min/max:", m.min(), m.max(), m.dtype)
        terrain_map_images.append(Image.fromarray(m, mode='RGB'))

    # Save images
    save_label = "all" if multi_video_training else video
    save_dir = os.path.join("results", save_label)
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(terrain_map_images):
        image.save(os.path.join(save_dir, f"{i:05d}.png"), "PNG")

    # Save video reconstructions of images...
    # images_to_video(save_dir, Path("video_recon") / f"{save_label}.mp4", frames_per_second=15)

    return

if __name__ == "__main__":
    main()

