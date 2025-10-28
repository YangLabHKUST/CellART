import numpy as np
import torch
import torch.utils.data as data
from .utils.io import load_array

class SSTDataset(data.Dataset):
    def __init__(self, manager):
        self.manager = manager
        # opt, logger, log_dir
        self.opt = manager.get_opt()
        self.logger = manager.get_logger()
        self.log_dir = manager.get_log_dir()
        # Gene Map data, Nuclei Mask data
        self.logger.info(f"Loading gene map from {self.opt.gene_map}")
        # self.gene_map = tifffile.imread(self.opt.gene_map)
        self.gene_map = load_array(self.opt.gene_map)
        self.logger.info(f"Gene map shape: {self.gene_map.shape}")
        self.logger.info(f"Loading nuclei mask from {self.opt.nuclei_mask}")
        if self.opt.nuclei_mask is None:
            self.nuclei_mask = np.ones(self.gene_map.shape[:2])
        else:
            self.nuclei_mask = load_array(self.opt.nuclei_mask)
        self.nuclei_mask = self.nuclei_mask.astype(np.int32)
        self.logger.info(
            f"Nuclei mask unique values: {len(np.unique(self.nuclei_mask))}, Shape: {self.nuclei_mask.shape}")

        # Assert the shape of gene_map and nuclei_mask_fp
        assert self.gene_map.shape[:2] == self.nuclei_mask.shape, "Gene map and nuclei mask shape mismatch"
        self.patch_size = self.opt.patch_size

        # Coords list of small patch
        h_starts = list(np.arange(0, self.gene_map.shape[0] - self.patch_size, self.patch_size))
        w_starts = list(np.arange(0, self.gene_map.shape[1] - self.patch_size, self.patch_size))
        h_starts.append(self.gene_map.shape[0] - self.patch_size)
        w_starts.append(self.gene_map.shape[1] - self.patch_size)

        self.coords_starts = [(x, y) for x in h_starts for y in w_starts]
        self.logger.info(f"Total {len(self.coords_starts)} patches of size {self.patch_size}x{self.patch_size}")
        # Shifting state: horizontal/vertical/off
        self.shifting_state = "off"
        self.new_seg_mask = None

    def get_coords_index(self, coords_h1, coords_w1):
        if self.shifting_state == "horizontal":
            return self.coords_starts.index((coords_h1, coords_w1 - self.patch_size // 4))
        elif self.shifting_state == "vertical":
            return self.coords_starts.index((coords_h1 - self.patch_size // 4, coords_w1))
        else:
            return self.coords_starts.index((coords_h1, coords_w1))

    def setting_new_segmentation(self, new_seg_mask):
        self.new_seg_mask = new_seg_mask

    def __len__(self):
        return len(self.coords_starts)

    def get_gene_num(self):
        return self.gene_map.shape[-1]

    def get_nuclei_shape(self):
        return self.nuclei_mask.shape

    def set_shifting_state(self, state):
        # Ensure the state is in ["horizontal", "vertical", "off"]
        assert state in ["horizontal", "vertical", "off"], "Shifting state must be horizontal/vertical/off"
        self.shifting_state = state


    def __getitem__(self, index):
        coords = self.coords_starts[index]
        coords_h1 = coords[0]
        coords_w1 = coords[1]
        if self.shifting_state == "horizontal":
            coords_w1 = coords_w1 + self.patch_size // 4
        elif self.shifting_state == "vertical":
            coords_h1 = coords_h1 + self.patch_size // 4
        coords_h2 = coords_h1 + self.patch_size
        coords_w2 = coords_w1 + self.patch_size

        expr = self.gene_map[coords_h1:coords_h2, coords_w1:coords_w2]

        if self.new_seg_mask is not None:
            nucl = self.new_seg_mask[coords_h1:coords_h2, coords_w1:coords_w2]
        else:
            nucl = self.nuclei_mask[coords_h1:coords_h2, coords_w1:coords_w2]

        # if the size is not patch_size x patch_size due to the edge, padding zero
        # expr: H*W*C, nucl: H*W
        if expr.shape[0] < self.patch_size:
            pad_h = self.patch_size - expr.shape[0]
            expr = np.pad(expr, ((0, pad_h), (0, 0), (0, 0)), 'constant', constant_values=0)
            nucl = np.pad(nucl, ((0, pad_h), (0, 0)), 'constant', constant_values=0)
        if expr.shape[1] < self.patch_size:
            pad_w = self.patch_size - expr.shape[1]
            expr = np.pad(expr, ((0, 0), (0, pad_w), (0, 0)), 'constant', constant_values=0)
            nucl = np.pad(nucl, ((0, 0), (0, pad_w)), 'constant', constant_values=0)

        expr_torch = torch.from_numpy(expr).permute(2, 0, 1)
        nucl_torch = torch.from_numpy(nucl)

        return expr_torch, nucl_torch, coords_h1, coords_w1


