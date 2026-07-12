import numpy as np
from PIL import Image
from functools import total_ordering
import glob
import os
import random
from terrain_classifier import gaussian_vote
from terrain_classifier import GAUSSIAN_PIXEL_STD_DEV_DENOM

GAUSSIAN_COLLECTION = True

# Integer terrain labels
@total_ordering
class Terrain():
    def __init__(self, rgb, ID, count):
        self.rgb = rgb
        self.ID = ID
        self.count = count

    def __eq__(self, other):
        if isinstance(other, Terrain):
            return self.count == other.count
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, Terrain):
            return self.count < other.count
        return NotImplemented



def integer_tiling(W, H, target):
    """Return an ideal near-square tile size and its image % coverage when in use. 

    Ideal tile size is defined by optimization of min(deviation from W and H + deviation from W = H).

    Args:
        W (int): width of image to tile
        H (int): height of image to tile
        target(int): desired length of tile side

    Returns: 
        tuple: (tile width, tile height, image coverage with use).
    """
    best = None
    min_penalty = float('inf')

    for cols in range(1, W+1):
        if W % cols == 0:
            w = W // cols
            for rows in range(1, H + 1):
                h = H // rows

                penalty = abs(w - target) + abs(h - target) + abs(w - h)

                if penalty < min_penalty:
                    min_penalty = penalty
                    best = (
                        w,
                        h,
                        (1 - (max(W%w,1) * max(H%h,1)) / (H * W)) # percentage coverage of tiling
                    )
    return best


def _collect_paired_paths(frame_dir, seg_dir):
    """Collect sorted PNG paths from a frame directory and its matching segmentation directory.

    Paths are sorted so ``zip(image_paths, segmentation_paths)`` pairs matching frames
    when filenames align across modalities.

    Args:
        frame_dir (str): Root directory containing frame PNGs (searched recursively).
        seg_dir (str): Root directory containing segmentation PNGs (searched recursively).

    Returns:
        tuple[list[str], list[str]]: ``(image_paths, segmentation_paths)``.
    """
    image_paths = sorted(glob.glob(os.path.join(frame_dir, "**/*.png"), recursive=True))
    segmentation_paths = sorted(glob.glob(os.path.join(seg_dir, "**/*.png"), recursive=True))
    return image_paths, segmentation_paths


def _video_names_under_parents(frames_parent, ann_parent, videos):
    """Resolve per-video subfolder names that exist under both frames and annotation parents.

    Args:
        frames_parent (str): Parent directory whose immediate subfolders are candidate videos.
        ann_parent (str): Parent directory for annotation trees (same subfolder names).
        videos (list[str] | None): If given, only these names are considered (in order).
            If None, every immediate subfolder of ``frames_parent`` that also exists under
            ``ann_parent`` is included.

    Returns:
        list[str]: Video subfolder names to process.
    """
    if videos is not None:
        candidates = list(videos)
    else:
        candidates = sorted(
            d for d in os.listdir(frames_parent)
            if os.path.isdir(os.path.join(frames_parent, d))
        )
    out = []
    for name in candidates:
        fp = os.path.join(frames_parent, name)
        ap = os.path.join(ann_parent, name)
        if os.path.isdir(fp) and os.path.isdir(ap):
            out.append(name)
    return out


def _resolve_train_test_video_sets(names, videos_train, videos_test):
    """Build train/test video name sets from optional explicit lists (multi-video mode).

    When only ``videos_test`` is set, every listed name in ``names`` is all-test; the rest
    are all-train. When only ``videos_train`` is set, the complement is all-test. When both
    are set, a name in both lists is treated as test; any name in ``names`` in neither list
    is assigned to train.

    Args:
        names (list[str]): Video subfolder names under consideration (preserves caller order).
        videos_train (list[str] | None): Subfolder names whose frames are all training.
        videos_test (list[str] | None): Subfolder names whose frames are all testing.

    Returns:
        tuple[set[str], set[str]]: ``(train_video_set, test_video_set)``, a partition of
        the set of ``names``.
    """
    names_set = set(names)
    explicit_test = set(videos_test or []) & names_set
    explicit_train = set(videos_train or []) & names_set
    explicit_train -= explicit_test
    unassigned = names_set - explicit_train - explicit_test
    if videos_train is None and videos_test is not None:
        final_test = explicit_test
        final_train = names_set - final_test
    elif videos_test is None and videos_train is not None:
        final_train = explicit_train
        final_test = names_set - final_train
    else:
        final_train = explicit_train | unassigned
        final_test = explicit_test
    return final_train, final_test


def _frame_level_train_test_split(
    image_paths,
    segmentation_paths,
    test_percentage,
    training_patch_size,
    training_stride,
    testing_patch_size,
    testing_stride,
    show_first_test,
):
    """Assign each frame independently to training (filtered patches) or testing (unfiltered).

    For each paired frame image and segmentation map, with probability ``test_percentage``
    the frame contributes unfiltered patches to testing; otherwise it contributes
    dominance-filtered patches to training.

    Args:
        image_paths (list[str]): Paths to frame PNG files.
        segmentation_paths (list[str]): Paths to matching segmentation PNG files.
        test_percentage (float): Probability in ``[0, 1]`` that a frame is held out for testing.
        training_patch_size (int): Patch side length for training extraction.
        training_stride (int): Stride for training patch tiling.
        testing_patch_size (int): Patch side length for testing extraction.
        testing_stride (int): Stride for testing patch tiling.
        show_first_test (bool): If True, display the first frame assigned to testing.

    Returns:
        tuple[list, list]: ``(labeled_training_images, labeled_testing_images)``, each a list
        of per-frame patch lists ``List[List[(PIL.Image, int), ...]]``.
    """
    labeled_training_images = []
    labeled_testing_images = []
    test_threshold = int(100 * test_percentage)
    test_num = 0
    train_num = 0
    for image, seg in zip(image_paths, segmentation_paths):
        if random.randrange(100) < test_threshold:
            labeled_testing_images.append(
                unfiltered_segmented_image_patch_extraction(
                    image, seg, testing_patch_size, testing_stride
                )
            )
            test_num += 1
            if show_first_test and test_num == 1:
                Image.open(image).show()
                Image.open(seg).show()
        else:
            labeled_training_images.append(
                filtered_segmented_image_patch_extraction(
                    image, seg, training_patch_size, training_stride
                )
            )
            train_num += 1
    if test_num + train_num > 0:
        print(f"full image ratio: {test_num/(test_num + train_num)}")
    return labeled_training_images, labeled_testing_images


def segmented_directory_patch_extraction(
    image_directory,
    segmentation_directory,
    multi_video_training,
    video_generalization,
    videos,
    test_percentage,
    training_patch_size,
    training_stride,
    testing_patch_size,
    testing_stride,
    videos_train=None,
    videos_test=None,
):
    """Build train/test patch datasets from segmented frame directories.

    Training patches use dominance filtering; testing patches keep all labels (unknown as 0).

    **Single video** (``multi_video_training=False``):``image_directory`` and
    ``segmentation_directory`` point to one video. ``video_generalization`` is ignored;
    each frame is randomly assigned to train or test using ``test_percentage``.

    **Multiple videos** (``multi_video_training=True``): both arguments are parents of
    per-video subfolders. If ``video_generalization=False``, the same per-frame random
    split is applied inside each video and results are concatenated. If
    ``video_generalization=True``, whole videos are assigned to train or test: either by
    random fraction ``test_percentage`` of the video count, or overridden by ``videos_train``
    and/or ``videos_test`` when at least one of those lists is non-None.

    Args:
        image_directory (str): Frame root for one video, or parent of video subfolders.
        segmentation_directory (str): Annotation root for one video, or parent if multi-video.
        multi_video_training (bool): If True, treat both roots as parents of video subfolders.
        video_generalization (bool): If True (and ``multi_video_training``), split by whole
            video; ignored when ``multi_video_training`` is False.
        videos (list[str] | None): When multi-video, restrict to these subfolder names; if
            None, use every subfolder present under both parents.
        test_percentage (float): Per-frame test probability when not using video-level split;
            when using video-level split without list overrides, fraction of videos assigned
            entirely to testing (``int(n_videos * test_percentage)``).
        training_patch_size (int): Training patch side length.
        training_stride (int): Training tiling stride.
        testing_patch_size (int): Testing patch side length.
        testing_stride (int): Testing tiling stride.
        videos_train (list[str] | None): Optional. If set (with multi-video and
            ``video_generalization``), these videos use only filtered training patches; see
            ``_resolve_train_test_video_sets`` for interaction with ``videos_test``.
        videos_test (list[str] | None): Optional. If set (with multi-video and
            ``video_generalization``), these videos use only unfiltered testing patches.

    Returns:
        tuple[list, list]: ``(labeled_training_images, labeled_testing_images)``, each
        ``List[List[(PIL.Image, int), ...]]`` grouped by source frame.
    """
    labeled_training_images = []
    labeled_testing_images = []

    # For training on frames from a single video 
    if not multi_video_training:
        image_paths, segmentation_paths = _collect_paired_paths(
            image_directory, segmentation_directory
        )
        labeled_training_batch, labeled_testing_batch = _frame_level_train_test_split(
            image_paths,
            segmentation_paths,
            test_percentage,
            training_patch_size,
            training_stride,
            testing_patch_size,
            testing_stride,
            show_first_test=True,
        )
        labeled_training_images = labeled_training_batch
        labeled_testing_images = labeled_testing_batch
        return labeled_training_images, labeled_testing_images

    # For training on frames from several videos
    frames_parent = image_directory
    ann_parent = segmentation_directory
    video_names = _video_names_under_parents(frames_parent, ann_parent, videos)
    # If empty directories
    if not video_names:
        return labeled_training_images, labeled_testing_images

    # For train-test split with video as smallest unit, collect videos and process into patches
    if video_generalization:
        override_lists = videos_train is not None or videos_test is not None
        # If specific split specified
        if override_lists:
            train_video_set, test_video_set = _resolve_train_test_video_sets(
                video_names, videos_train, videos_test
            )
        else: # Otherwise, random choice
            names_shuffled = list(video_names)
            random.shuffle(names_shuffled)
            n_test = int(len(names_shuffled) * test_percentage)
            n_test = max(0, min(len(names_shuffled), n_test))
            test_video_set = set(names_shuffled[:n_test])
            train_video_set = set(names_shuffled[n_test:])

        shown_first_test = False

        # Collect images from both video directories and process into patches
        for video in video_names:
            frame_dir = os.path.join(frames_parent, video)
            seg_dir = os.path.join(ann_parent, video)
            image_paths, segmentation_paths = _collect_paired_paths(frame_dir, seg_dir)
            # Process testing videos
            if video in test_video_set:
                for image, seg in zip(image_paths, segmentation_paths):
                    labeled_testing_images.append(
                        unfiltered_segmented_image_patch_extraction(
                            image, seg, testing_patch_size, testing_stride
                        )
                    )
                    # Display for debugging
                    if not shown_first_test:
                        Image.open(image).show()
                        Image.open(seg).show()
                        shown_first_test = True
            else: # Process training videos
                for image, seg in zip(image_paths, segmentation_paths):
                    labeled_training_images.append(
                        filtered_segmented_image_patch_extraction(
                            image, seg, training_patch_size, training_stride
                        )
                    )
        n_vid = len(video_names)
        print(
            f"video-level split: {len(test_video_set)}/{n_vid} videos -> test, "
            f"{len(train_video_set)}/{n_vid} -> train"
        )
        return labeled_training_images, labeled_testing_images

    # For train-test split with frame as smallest unit, collect frames and process into patches    
    for video_index, video in enumerate(video_names):
        frame_dir = os.path.join(frames_parent, video)
        seg_dir = os.path.join(ann_parent, video)
        image_paths, segmentation_paths = _collect_paired_paths(frame_dir, seg_dir)
        labeled_training_batch, labeled_testing_batch = _frame_level_train_test_split(
            image_paths,
            segmentation_paths,
            test_percentage,
            training_patch_size,
            training_stride,
            testing_patch_size,
            testing_stride,
            show_first_test=(video_index == 0),
        )
        labeled_training_images.extend(labeled_training_batch)
        labeled_testing_images.extend(labeled_testing_batch)

    return labeled_training_images, labeled_testing_images


def unsegmented_directory_patch_extraction(image_directory, patch_size, stride):
    """Converts images into a list per-full-image lists of 64x64 image-patches.

    Does not filter patches in any way.

    Args:
        image_directory (string): Relative file path of directory with images.

    Returns:
        list(list(Image)): List of per-full-image lists of images, where each is represented as tiling patches.
    
    """
    
    # Collect files recursively and sort by name for deterministic output
    image_paths = sorted(glob.glob(os.path.join(image_directory, "**/*.png"), recursive=True))
    
    # Store each patch-array image into images
    images = []
    for image in image_paths:
        images.append(unsegmented_image_patch_extraction(image, patch_size, stride))
    
    return images

def filtered_segmented_image_patch_extraction(image_path, segmentation_path, patch_size, stride):
    """Converts image into a list of labeled, tiling image-patches.

    Filters patches such that all are dominated by a monitored terrain type.

    Args:
        image_path (string): Relative file path of images.
        segmentation_path (string): Relative file path of pixel-annotated image.

    Returns:
        list(Image, int): List of patch-label pairs.
    
    """
    
    # Parse image
    im = Image.open(image_path)

    # Convert to greymap with standard Luma transform
    im = im.convert("L")

    # Normalize Gamma
    gamma = 2.2
    im = im.point(lambda i: pow((i/255), 1/gamma) * 255)

    # Parse segmentation map
    seg = Image.open(segmentation_path)

    # Check segmentation map
    #seg.show()

    # Check image
    #im.show()


    patches = []
    seg_patches = []
    for y_upper in range(0, im.height - patch_size + 1, stride):
        for x_left in range(0, im.width - patch_size + 1, stride):
            patch = im.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            patches.append(patch)
            seg_patch = seg.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            seg_patches.append(seg_patch)


    ### View a patch
    #i = 7
    #patches[i].show()
    #seg_patches[i].show()


    # Surveying the following terrain labels
    dirt = Terrain((108, 64, 20), 1, 0)
    grass = Terrain((0, 102, 0), 2, 0)
    asphalt = Terrain((64, 64, 64), 3, 0)
    gravel = Terrain((255, 128, 0), 4, 0)


    # Save patches that are dominated (>60%) by one valid (above) terrain types
    dominated_patches = []
    # How much of the patch must be known to include it in training
    threshold = 0.6
    # Not really a minibatch, misnomer.
    mini_batch_percent = 0.30
    gauss_mask = gaussian_vote(patch_size, patch_size/GAUSSIAN_PIXEL_STD_DEV_DENOM)
    for patch, seg_patch in zip(patches, seg_patches):

        # Convert patch into np array
        patch_array = np.array(seg_patch)

        if GAUSSIAN_COLLECTION:
        # Apply terrain masks and count gaussian-weighted non-zero values for each
            dirt.count    = (np.all(patch_array == dirt.rgb, axis=-1) * gauss_mask).sum()
            grass.count   = (np.all(patch_array == grass.rgb, axis=-1) * gauss_mask).sum()
            asphalt.count = (np.all(patch_array == asphalt.rgb, axis=-1) * gauss_mask).sum()
            gravel.count  = (np.all(patch_array == gravel.rgb, axis=-1) * gauss_mask).sum()
            largest_terrain = max(dirt, grass, asphalt, gravel)
            # Define peak based on gaussian
            if (largest_terrain.count/((np.full((patch_size, patch_size), 1) * gauss_mask).sum()) >= threshold and np.random.rand() < mini_batch_percent):
                dominated_patches.append((patch, largest_terrain.ID))
        else:  # Apply terrain masks and count non-zero values for each
            dirt.count    = np.count_nonzero(np.all(patch_array == dirt.rgb, axis=-1))
            grass.count   = np.count_nonzero(np.all(patch_array == grass.rgb, axis=-1))
            asphalt.count = np.count_nonzero(np.all(patch_array == asphalt.rgb, axis=-1))
            gravel.count  = np.count_nonzero(np.all(patch_array == gravel.rgb, axis=-1))
            largest_terrain = max(dirt, grass, asphalt, gravel)
            if (largest_terrain.count/(patch_size * patch_size) >= threshold):
                dominated_patches.append((patch, largest_terrain.ID))

            
        # Check terrain spread
        #print(f"Dirt: {dirt.count}, Grass: {grass.count}, asphalt: {asphalt.count}, gravel: {gravel.count}")



    ### View all dominatedpatches
    #for img in dominated_patches: 
    #    img[0].show()

    return dominated_patches



def unfiltered_segmented_image_patch_extraction(image_path, segmentation_path, patch_size, stride):
    """Converts image into a list of labeled, tiling image-patches.

    All patches with majority known terrains are kept. For patching test data.

    Args:
        image_path (string): Relative file path of images.
        segmentation_path (string): Relative file path of pixel-annotated image.

    Returns:
        list(Image, int): List of patch-label pairs.
    
    """
    
    # Parse image
    im = Image.open(image_path)

    # Convert to greymap with standard Luma transform
    im = im.convert("L")

    # Normalize Gamma
    gamma = 2.2
    im = im.point(lambda i: pow((i/255), 1/gamma) * 255)

    # Parse segmentation map
    seg = Image.open(segmentation_path)

    # How much of patch must be known for it to vote on frame pixels
    threshold = 0.9
    patches = []
    seg_patches = []
    for y_upper in range(0, im.height - patch_size + 1, stride):
        for x_left in range(0, im.width - patch_size + 1, stride):
            patch = im.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            patches.append(patch)
            seg_patch = seg.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            seg_patches.append(seg_patch)

    # Surveying the following terrain labels
    dirt = Terrain((108, 64, 20), 1, 0)
    grass = Terrain((0, 102, 0), 2, 0)
    asphalt = Terrain((64, 64, 64), 3, 0)
    gravel = Terrain((255, 128, 0), 4, 0)


    # Save patches
    terrain_patches = []
    gauss_mask = gaussian_vote(patch_size, patch_size/GAUSSIAN_PIXEL_STD_DEV_DENOM)

    for patch, seg_patch in zip(patches, seg_patches):

        # Convert patch into np array
        patch_array = np.array(seg_patch)

        if GAUSSIAN_COLLECTION:
        # Apply terrain masks and count gaussian-weighted non-zero values for each
            dirt.count    = (np.all(patch_array == dirt.rgb, axis=-1) * gauss_mask).sum()
            grass.count   = (np.all(patch_array == grass.rgb, axis=-1) * gauss_mask).sum()
            asphalt.count = (np.all(patch_array == asphalt.rgb, axis=-1) * gauss_mask).sum()
            gravel.count  = (np.all(patch_array == gravel.rgb, axis=-1) * gauss_mask).sum()
            largest_terrain = max(dirt, grass, asphalt, gravel)
            total_terrain = dirt.count + grass.count + asphalt.count + gravel.count
            # Define peak based on gaussian
            if (total_terrain/((np.full((patch_size, patch_size), 1) * gauss_mask).sum()) >= threshold):
                terrain_patches.append((patch, largest_terrain.ID))
            else:
                terrain_patches.append((patch, 0))
        else:  # Apply terrain masks and count non-zero values for each
            dirt.count    = np.count_nonzero(np.all(patch_array == dirt.rgb, axis=-1))
            grass.count   = np.count_nonzero(np.all(patch_array == grass.rgb, axis=-1))
            asphalt.count = np.count_nonzero(np.all(patch_array == asphalt.rgb, axis=-1))
            gravel.count  = np.count_nonzero(np.all(patch_array == gravel.rgb, axis=-1))
            largest_terrain = max(dirt, grass, asphalt, gravel)
            total_terrain = dirt.count + grass.count + asphalt.count + gravel.count
            if (total_terrain/(patch_size * patch_size) >= threshold):
                terrain_patches.append((patch, largest_terrain.ID))
            else:
                terrain_patches.append((patch, 0))
                

        # Check terrain spread
        #print(f"Dirt: {dirt.count}, Grass: {grass.count}, asphalt: {asphalt.count}, gravel: {gravel.count}")

    return terrain_patches



def unsegmented_image_patch_extraction(image, patch_size, stride):
    """Converts image into a list of tiling image-patches. 
    
    Does not filter patches in any way.

    Args:
        image_path (string): Relative file path of images.

    Returns:
        list(Image): List of patches.
    
    """

    # Parse image
    im = Image.open(image)

    # Convert to greymap with standard Luma transform
    im = im.convert("L")

    # Normalize Gamma
    gamma = 2.2
    im = im.point(lambda i: pow((i/255), 1/gamma) * 255)

    patches = []
    for y_upper in range(0, im.height - patch_size + 1, stride):
        for x_left in range(0, im.width - patch_size + 1, stride):
            patch = im.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            patches.append(patch)

    return patches

