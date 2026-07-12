import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def extract_feature_vectors(images, cell_size, num_neighbors, pixel_radius):
    """Extracts LBP histogram features for each image by tiling into cells and concatenating.

    The patch is split into a square grid of ``cell_size``-by-``cell_size`` regions (e.g. a
    64x64 patch with ``cell_size == 16`` gives a 4x4 grid, 16 cells, each 16x16 = 256 pixels).
    LBP is computed once on the **whole** patch so each pixel's pattern uses the correct
    neighbors, including those in adjacent cells. Per-cell LBP values are then histogrammed
    and the histograms are concatenated left-to-right, top-to-bottom.

    Args:
        images (list): PIL grayscale images (square patches).
        cell_size (int): Side length in pixels of each grid cell; patch side must be divisible.

    Returns:
        np.ndarray: Shape ``(len(images), feature_dimension)``, one feature vector per image.
    """


    features = np.array(Parallel(n_jobs=-1)(
        delayed(extract_feature_vector)(image, cell_size, num_neighbors, pixel_radius) for image in images
    ), dtype=np.float32)

    return features


def extract_feature_vector(image, cell_size, num_neighbors, pixel_radius):
    """Extracts LBP histogram features for an image by tiling into cells and concatenating.

    The patch is split into a square grid of ``cell_size``-by-``cell_size`` regions (e.g. a
    64x64 patch with ``cell_size == 16`` gives a 4x4 grid, 16 cells, each 16x16 = 256 pixels).
    LBP is computed once on the **whole** patch so each pixel's pattern uses the correct
    neighbors, including those in adjacent cells. Per-cell LBP values are then histogrammed
    and the histograms are concatenated left-to-right, top-to-bottom.

    Args:
        image (PIL.Image): a square PIL image.
        cell_size (int): Side length in pixels of each grid cell; patch side must be divisible.

    Returns:
        np.ndarray: Shape ``(feature_dimension)``, one feature vector.
    """

    # LBP Configuration
    num_neighbors = 8
    pixel_radius = 1
    method = "nri_uniform"

    # Histogram bin count for "nri_uniform" labels with P neighbors
    histogram_bin_count = num_neighbors * (num_neighbors - 1) + 3

    # Check for square patches and patch-tiling cell_size
    patch_side_pixels = image.size[0]
    cells_per_axis = patch_side_pixels // cell_size
    if patch_side_pixels != image.size[1]:
        raise ValueError("LBP extractor assumes square patches.")
    if cells_per_axis * cell_size != patch_side_pixels:
        raise ValueError(
            f"Patch side {patch_side_pixels} is not divisible by cell_size {cell_size}."
        )

    # Depreciated
    #num_cells = cells_per_axis * cells_per_axis
    #feature_dimension = num_cells * histogram_bin_count
    #feature_matrix = np.zeros((feature_dimension), dtype=np.float64)

    # Convert PIL image to array of ints in [0, 255].
    patch = np.array(image).astype(np.int32)

    # One LBP map for the full patch: boundary handling is shared across the cell grid.
    lbp_label_map = local_binary_pattern(
        patch, num_neighbors, pixel_radius, method=method
    )

    per_cell_histograms = []

    # Form histograms per-cell
    for cell_row_idx in range(cells_per_axis):
        for cell_col_idx in range(cells_per_axis):
            row_start = cell_row_idx * cell_size
            col_start = cell_col_idx * cell_size
            
            lbp_in_cell = lbp_label_map[
                row_start : row_start + cell_size,
                col_start : col_start + cell_size,
            ]

            cell_histogram, _ = np.histogram(
                lbp_in_cell.ravel(),
                bins=histogram_bin_count,
                range=(0, histogram_bin_count),
            )

            # Normalize each cell so scale does not depend on cell area.
            total_count = float(np.sum(cell_histogram))
            if total_count > 0.0:
                cell_histogram = cell_histogram / total_count

            per_cell_histograms.append(cell_histogram)

    features = np.concatenate(per_cell_histograms, axis=0)

    return features
