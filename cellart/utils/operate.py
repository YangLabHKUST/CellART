import cv2
import numpy as np
import torch
DEFAULT_CHUNK_SIZE = 1000

def get_cell_size(mask, chunk_size = DEFAULT_CHUNK_SIZE):
    """
    Get the size of each cell in the mask.

    Args:
        mask (torch.Tensor): Input segmentation mask of shape (H, W), where each unique value represents a cell.

    Returns:
        torch.Tensor: Cell sizes of shape (C,), where C is the number of cells.
    """
    unique_cells = torch.unique(mask)
    if unique_cells[0] == 0:
        unique_cells = unique_cells[1:]

    cell_sizes = torch.zeros(unique_cells.size(0), device=mask.device, dtype=mask.dtype)
    for i in range(0, unique_cells.size(0), chunk_size):
        low = i
        up = min(i + chunk_size, unique_cells.size(0))
        cell_sizes[low:up] = (mask[None, :, :] == unique_cells[low:up, None, None]).sum((1, 2))

    return cell_sizes

def get_cell_location(mask, chunk_size = DEFAULT_CHUNK_SIZE):
    unique_cells = torch.unique(mask)
    if unique_cells[0] == 0:
        unique_cells = unique_cells[1:]

    cell_locations = torch.zeros((unique_cells.size(0), 2), device=mask.device, dtype=mask.dtype)
    for i in range(0, unique_cells.size(0), chunk_size):
        low = i
        up = min(i + chunk_size, unique_cells.size(0))
        per_cell_mask = split_mask_chunk(mask, low, up)
        cell_location_x = torch.stack([torch.nonzero(per_cell_mask[i], as_tuple=True)[0].float().mean() for i in
                                       range(per_cell_mask.shape[0])])
        cell_location_y = torch.stack([torch.nonzero(per_cell_mask[i], as_tuple=True)[1].float().mean() for i in
                                       range(per_cell_mask.shape[0])])
        cell_locations[low:up] = torch.stack([cell_location_x, cell_location_y], dim=1)

    return cell_locations

def split_mask_chunk(mask, low, up):
    """
    Split the mask into individual binary masks for each cell.

    Args:
        mask (torch.Tensor): Input segmentation mask of shape (H, W), where each unique value represents a cell.
        low (int): Lower bound of the cell IDs to process.
        up (int): Upper bound of the cell IDs to process.

    Returns:
        torch.Tensor: Binary masks of shape (C, H, W), where C is the number of cells.
    """
    unique_cells = torch.unique(mask)
    if unique_cells[0] == 0:
        unique_cells = unique_cells[1:]
    binary_masks_chunk = (mask[None, :, :] == unique_cells[low:up, None, None]).float()
    return binary_masks_chunk

def aggregate_gene_expression(mask, gene_map, method='sum'):
    """
    Aggregate gene expression per cell by summing up all gene expressions inside each cell's mask.

    Args:
        mask (torch.Tensor): Input segmentation mask of shape (H, W), where each unique value represents a cell.
        gene_map (torch.Tensor): Gene expression map of shape (H, W, G), where G is the number of genes.

    Returns:
        torch.Tensor: Cell-level gene expression matrix of shape (C, G), where C is the number of cells.
    """
    # Ensure both inputs are on the GPU
    mask = mask.to('cuda') if not mask.is_cuda else mask
    gene_map = gene_map.to('cuda') if not gene_map.is_cuda else gene_map

    # Flatten spatial dimensions (H, W) to a single dimension
    H, W, G = gene_map.shape
    flattened_mask = mask.view(-1)  # Shape: (H*W,)
    flattened_genes = gene_map.view(-1, G)  # Shape: (H*W, G)

    # Get unique cell IDs and a mapping from pixels to cell indices
    unique_cells, inverse_indices = torch.unique(flattened_mask, return_inverse=True)

    # If background (0) is present, exclude it
    if unique_cells[0] == 0:
        unique_cells = unique_cells[1:]
        valid_mask = (flattened_mask != 0)
        flattened_genes = flattened_genes[valid_mask]
        inverse_indices = inverse_indices[valid_mask] - 1  # Adjust indices to remove background cell

    # Aggregate gene expressions using scatter_add
    C = unique_cells.size(0)  # Number of cells
    aggregated_expression = torch.zeros((C, G), device=gene_map.device, dtype=gene_map.dtype)
    aggregated_expression = aggregated_expression.index_add(0, inverse_indices, flattened_genes)

    if method == 'mean':
        # Count the number of pixels in each cell
        # First split the mask into individual binary masks for each cell
        # binary_masks = split_mask(mask)
        # cell_sizes = binary_masks.sum((1, 2))
        cell_sizes = get_cell_size(mask)
        # Compute the mean gene expression per cell
        aggregated_expression /= cell_sizes[:, None]

    return aggregated_expression


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_softmask_fast(cell_nuclei_mask, tau=5):
    # Convert mask to uint8
    binary_mask = (cell_nuclei_mask > 0).astype(np.uint8)

    # Compute distance transform (distance to nearest zero pixel)
    dist_transform = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)

    # Apply sigmoid function
    soft_mask = sigmoid(dist_transform / tau)

    return 1 - soft_mask