import numpy as np
from scipy.signal import convolve2d
from scipy.linalg import norm as l2_norm
from joblib import Parallel, delayed

def l2_hys_norm(v, threshold=0.2):
    # Find L2 norm and normalize v
    norm = l2_norm(v)
    if norm > 0:
        v = v/norm

    # Apply hystersis (bound to interval(0, threshold))
    v = np.clip(v, 0, threshold)

    # Re-normalize
    norm =  l2_norm(v)
    if norm > 0:
        v = v/norm

    return v
    

def extract_feature_vectors(images, block_size, cell_size):
    """Extracts HOG features for each passed in image in images 
    
    Assumes square images, blocks, and cells
    
    Args:
        images (List(Image)): Images to extract from.
        block_size (int): Length of square normalization block side.
        cell_size (int): Length of square cell side.

    Returns:
        np.array[image]: list of per-image normalized numerical array created by concatenating all the normalized cell histograms from all overlapping black across the entire image.
    
    """

    features = np.array(Parallel(n_jobs=-1)(
        delayed(extract_feature_vector)(image, block_size, cell_size) for image in images
    ), dtype=np.float32)


    return features


def extract_feature_vector(image, block_size, cell_size):
    """Extracts HOG features for a passed in image
    
    Assumes square images, blocks, and cells
    
    Args:
        images (PIL.Image): Image to extract from.
        block_size (int): Length of square normalization block side.
        cell_size (int): Length of square cell side.

    Returns:
        np.array[cell_histograms]: array created by concatenating all the normalized cell histograms from all overlapping black across the entire image.
    
    """

    image_size = image.size[0]

    # Define gradient kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1] 
        ])
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1] 
        ])
    
    # Histogram variables
    bins = 9
    bin_size = 180/bins

    # Cell-width of an image
    sqrt_num_cells = int(image_size / cell_size)
    # Block-width of an image
    sqrt_num_blocks = (sqrt_num_cells - (block_size - 1))

    # Dimension of Feature Vectors = (blocks per image) * (cells per block) * (histogram bins per cell)
    feature_dimension = (sqrt_num_blocks * sqrt_num_blocks) * (block_size * block_size) * bins
    
    features = np.zeros((feature_dimension), dtype=float)

    # Convert image to np array of floats
    image = np.array(image).astype(np.float32) / 255.0
    
    # Add symmetric padding
    image = np.pad(image, 1, mode='reflect')

    # Compute Gradients
    G_x = convolve2d(image, sobel_x, mode='valid')
    G_y = convolve2d(image, sobel_y, mode='valid')

    mag = np.hypot(G_x, G_y)
    ori = np.atan2(G_y, G_x)

    # Split image into cell_size x cell_size cells
    # "Tiles" the array in view only, no memory copying. (image_size/cell_size) x (image_size/cell_size) grid of (cell_size) x (cell_size) cells.
    mag_cells = mag.reshape(sqrt_num_cells, sqrt_num_cells, cell_size, cell_size).transpose(0, 2, 1, 3)
    ori_cells = ori.reshape(sqrt_num_cells, sqrt_num_cells, cell_size, cell_size).transpose(0, 2, 1, 3)

    # Flattens the cell view grid into a list of cells
    mag_cells = mag_cells.reshape(sqrt_num_cells * sqrt_num_cells, cell_size, cell_size)
    ori_cells = ori_cells.reshape(sqrt_num_cells * sqrt_num_cells, cell_size, cell_size)

    # Compute histograms for each cell
    cell_histograms = np.zeros((sqrt_num_cells * sqrt_num_cells, bins), dtype=float)
    for cell_idx, (mag_cell, ori_cell) in enumerate(zip(mag_cells, ori_cells)):
        hog = np.zeros(bins)
        for mag, ori in zip(mag_cell.ravel(), ori_cell.ravel()):
            # Map [0,2pi]->[0,180)s
            theta = (np.degrees(ori) % 180)
            # Linearly interpolate magnitude between nearest bins
            bin_float = theta / bin_size
            lower_bin = int(bin_float) % bins
            upper_bin = (lower_bin + 1) % bins

            hog[upper_bin] += mag * (bin_float - lower_bin)
            hog[lower_bin] += mag * (upper_bin - bin_float)

        # Add cell histogram to larger array  
        cell_histograms[cell_idx] = hog

    # Reshape cell-histograms into grid  -> (cell_x, cell_y, histogram) 
    cell_histograms = cell_histograms.reshape(cell_size, cell_size, bins)

    block_histograms = []
    # Apply L2-Hys norm to each block
    for i in range(sqrt_num_blocks):
        for j in range(sqrt_num_blocks):
            block = cell_histograms[i:i+block_size, j:j+block_size].flatten()
            normalized_block = l2_hys_norm(block)
            block_histograms.extend(normalized_block)
                                    

    return np.array(block_histograms)
    