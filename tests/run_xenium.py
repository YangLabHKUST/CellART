import cellart
from pathlib import Path
import wandb
import os

manager = cellart.ExperimentManager(
    # Basic settings of data (Must be specified)
    gene_map = "/import/home2/yhchenmath/Code/Triplet/Data/XeniumBreastCancer/gene_map.npy",
    nuclei_mask = "/import/home2/yhchenmath/Code/Triplet/Data/XeniumBreastCancer/segmentation_mask.npy",
    basis = "/import/home2/yhchenmath/Code/Triplet/Data/XeniumBreastCancer_one_tumor/basis.npy",
    gene_names = "/import/home2/yhchenmath/Code/Triplet/Data/XeniumBreastCancer/filtered_gene_names.txt",
    log_dir = "/import/home2/yhchenmath/Code/Triplet/LOG/TestCellART004",
    celltype_names = "/import/home2/yhchenmath/Code/Triplet/Data/XeniumBreastCancer_one_tumor/celltype_names.txt",
    # Training parameters
    epoch = 200,
    seg_training_epochs = 5,
    deconv_warmup_epochs = 100,
    pred_period = 50,
    gpu = "0"
)

# Update options
opt = manager.get_opt()
# Print options (Namespace)
print(opt)

# Set up wandb for logging and visualization
run = wandb.init(project="CellART", dir=manager.get_log_dir(), config=opt,
                    name=os.path.basename(os.path.normpath(manager.get_log_dir())))

# Set up dataset
# dataset = cellart.SSTDataset(manager)
# gene_map_shape = dataset.gene_map.shape

# model = cellart.CellARTModel(manager, gene_map_shape, len(dataset.coords_starts))
# model.train_model(dataset)
