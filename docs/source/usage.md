# Usage: Quick Start Guide for CellART

This comprehensive guide provides a detailed explanation of how to preprocess and run CellART, a framework for extracting single-cell information from high-resolution spatial transcriptomics (ST) datasets. 

You can check the examples of running CellART on 


## Preprocessing

### Input Requirements
For CellART, the following inputs are necessary:

1. **Raw Data from Spatial Transcriptomics Platforms**:
   - **Spatial Transcriptomics Data**: Typically provided as a table containing transcript information (gene counts) and their spatial coordinates.
   - **Staining Images**: DAPI or H&E images, which need to be aligned with the transcriptomics data during preprocessing.

2. **scRNA-seq Reference Dataset**:
   - A reference dataset containing cell type annotations paired with the ST data.

### Standardized Input Format
After preprocessing, the raw data is converted into the following standardized format required by CellART:
- **Gene Map**: A 3D array of shape `H×W×G`, where `H` and `W` are the spatial dimensions, and `G` is the number of unique genes. For platforms such as **VisiumHD**, highly variable genes (HVGs) are selected from the scRNA-seq reference. For other platforms, the intersection of genes between ST and scRNA datasets is used. The resolution is typically set to 1 or 2 µm.
- **Nuclei Segmentation Mask**: A 2D array of shape `H×W`, where each pixel is labeled as `0` for background or assigned a unique cell ID. This mask can be generated using tools such as **Cellpose** or **StarDist**, or directly obtained from the raw dataset.
- **Basis Matrix**: A matrix of shape `C×G`, where `C` is the number of cell types and `G` is the number of genes. This matrix describes the gene expression signatures for each cell type and is derived from the scRNA-seq reference.



## Platform-Specific Preprocessing

### Xenium
The following example demonstrates how to preprocess Xenium data for CellART:
```python
from cellart.utils.preprocess import SingleCellPreprocessor, XeniumPreprocessor
from cellart.utils.io import load_list
import scanpy as sc

# Processed data save dir
save_dir = '/{YOUR_PATH}/Xenium/'
# Transcripts and nucleus boundary files in data directory
transcripts_file = "/{YOUR_DATA_PATH}/transcripts.parquet"
nucleus_boundary_10X = "/{YOUR_DATA_PATH}/nucleus_boundaries.parquet"

st_preprocessor = XeniumPreprocessor(transcripts_file, nucleus_boundary_10X, save_dir)

# Annotated scRNA reference path
sc_adata = sc.read("/{YOUR_SC_DATA}/sc_data.h5ad")
# Remember to specific your celltype_col and make sure your are using raw count data
sc_preprocessor = SingleCellPreprocessor(sc_adata, celltype_col = "celltype", save_path= save_dir, st_gene_list=load_list(save_dir + "/st_gene_list.txt"))

# If you are using Xenium 5K, please using hvg options like in VisiumHD
sc_preprocessor.preprocess()

st_preprocessor.prepare_sst(load_list(save_dir + "/filtered_gene_names.txt"))
st_preprocessor.get_nuclei_segmentation()
```



---

### VisiumHD
For VisiumHD, the following example demonstrates how to preprocess data:
```python
from cellart.utils.preprocess import SingleCellPreprocessor, VisiumHDPreprocessor
from cellart.utils.io import load_list
import scanpy as sc

# Processed data save dir
save_dir = '/{YOUR_PATH}/VisiumHD/'
# Path to 002um spot data
path = "/{YOUR_DATA_PATH}/square_002um/"
# Path to he
source_image_path = "/{YOUR_DATA_PATH}/XXX_tissue_image.tif" # or btf
# Path to spatial dir
spaceranger_image_path = "/{YOUR_DATA_PATH}/spatial"

st_preprocessor = VisiumHDPreprocessor(path, source_image_path, spaceranger_image_path, save_dir)
st_preprocessor.get_nuclei_segmentation()

# Annotated scRNA reference path
sc_adata = sc.read("/{YOUR_SC_DATA}/sc_data.h5ad")
# Remember to specific your celltype_col and make sure your are using raw count data
sc_preprocessor = SingleCellPreprocessor(sc_adata, celltype_col = "celltype", save_path= save_dir, st_gene_list=load_list(save_dir + "/st_gene_list.txt"))

# Selecting highly variable genes 2000 / 3000
sc_preprocessor.preprocess(hvg_method="seurat_v3", n_hvg=3000)

st_preprocessor.prepare_sst(load_list(save_dir + "/filtered_gene_names.txt"))
```

---

### Stereoseq
For Stereoseq, the following steps illustrate how to preprocess data:
```python
# Stereoseq need to use the package to read the raw data
import stereo

gef_data = stereo.io.read_gef("/{YOUR_DATA_PATH}/XXX.raw.gef", bin_size=4) # 2um resolution
adata = stereo.io.stereo_to_anndata(gef_data)
# You may need additional step to map the ensembl_id to gene name in the reference
adata.write_h5ad("/{YOUR_DATA_PATH}/bin4.h5ad")

from cellart.utils.preprocess import SingleCellPreprocessor
import scanpy as sc
import tifffile
import cv2
import scipy
from scipy.sparse import coo_matrix
import numpy as np
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

img = tifffile.imread("/{YOUR_DATA_PATH}/XXX_ssDNA_regist.tif")
adata = sc.read_h5ad("/{YOUR_DATA_PATH}/bin4.h5ad")

# Processed data save dir
save_dir = '/{YOUR_PATH}/Stereoseq/'


# Match the 2 um resolution
resized = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
# Cellpose to segment the nuclei
def segment_dapi(img, diameter=None, use_cpu=False):
     """Segment nuclei in DAPI image using Cellpose"""
     use_gpu = True if not use_cpu else False
     model = models.Cellpose(gpu=use_gpu, model_type="cyto")
     channels = [0, 0]
     mask, _, _, _ = model.eval(img, diameter=diameter, channels=channels)
     print(mask.max())
     return mask
# Adjust the diameter to get a better cellpose result
nuclei_seg = segment_dapi(resized, diameter=2)
# Save nuclei segmentation
nuclei_seg = np.save("/import/home3/yhchenmath/Dataset/DeconvSeg/stereoseq_mouse_brain/nuclei_seg.npy")

# Annotated scRNA reference path
sc_adata = sc.read("/{YOUR_SC_DATA}/sc_data.h5ad")
# Select genes in sc_adata that are also in st data
common_genes = sc_adata.var_names.intersection(adata.var_names)
sc_preprocessor = SingleCellPreprocessor(sc_adata, celltype_col = "celltype", save_path= save_dir, st_gene_list=common_genes)

sc_preprocessor.preprocess(hvg_method="seurat_v3", n_hvg=3000)

# Process to get gene map
hvg_genes = pd.read_csv("/{YOUR_SAVE_DIR}/filtered_gene_names.txt", header=None)
# Tolist
hvg_genes = hvg_genes[0].tolist()
# Select
adata_hvg = adata[:, hvg_genes].copy()
# Filter 
sc.pp.filter_cells(adata_hvg, min_genes=1)

adata_st = adata_hvg.copy()
# x//4, y//4 to fit the resolution
adata_st.obs['x'] = adata_st.obs['x'] // 4
adata_st.obs['y'] = adata_st.obs['y'] // 4
adata_st.var_names_make_unique()
map_width = img.shape[0] // 4
map_height = img.shape[1] // 4
adata_st.obs_names_make_unique()


def process_gene_chunk(adata_sub, temp_dir, map_height, map_width):
    for gene in list(adata_sub.var_names):
        temp_df = adata_sub.obs[['x', 'y']]
        if scipy.sparse.issparse(adata_sub.X):
            temp_df[gene] = adata_sub[:, gene].X.toarray().flatten()
        else:
            temp_df[gene] = adata_sub[:, gene].X.flatten()
        map = coo_matrix((temp_df[gene], (temp_df['x'], temp_df['y'])), shape=(map_height, map_width)).toarray()
        np.save(os.path.join(temp_dir, gene + ".npy"), map)
        print(f"Gene {gene} processed.")

# Number of multiprocess
n_processes = 25
filtered_gene_list = hvg_genes
gene_names_chunks = np.array_split(filtered_gene_list, n_processes)
processes = []

temp_dir = os.path.join("/PATH/per_gene_map") # A temp dir to store temp data
os.makedirs(temp_dir, exist_ok=True)

for i, gene_chunk in enumerate(gene_names_chunks):
    adata_sub = adata_st[:, gene_chunk].copy()
    p = mp.Process(target=process_gene_chunk, args= (adata_sub, temp_dir, map_height, map_width))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# Combine channel-wise
from tqdm.notebook import tqdm
gene_map = None
for i, gene in tqdm(enumerate(filtered_gene_list), total=len(filtered_gene_list)):
    map = np.load(os.path.join(temp_dir, gene + ".npy"))
    if gene_map is None:
        gene_map = np.zeros((map.shape[0], map.shape[1], len(filtered_gene_list)), dtype=np.uint8)
        gene_map[:, :, i] = map
    else:
        gene_map[:, :, i] = map

# Save the combined map
np.save("/{YOUR_SAVE_DIR}/gene_map.npy", gene_map)
```
---

### MERFISH


The transcripts.csv file can be convert to Xenium-like format and then processed in the same way (simply rename the column). The image can be matched and use Cellpose to segment like Stereoseq. You can also directly use the nuclei boundaries from platform, details can follow the official tutorials for different versions of MERFISH. 


## Run CellART

After preprocessing (or manually converting raw data into the required format), you will have the following files in your save directory (`save_dir` specified in the preprocessing code):

- **gene_map.npy**: A [HxWxG numpy file], where each element represents the gene count at a specific spatial location.
- **segmentation_mask.npy**: A [HxW numpy file] containing the nuclei segmentation mask.
- **basis.npy**: A [CxG numpy file] representing the cell type signature matrix derived from the scRNA reference.
- **filtered_gene_names.txt**: A text file listing the gene names corresponding to the order of the `G` channel in `gene_map.npy` and `basis.npy`.
- **celltype_names.txt**: A text file listing the cell type names matching the order of the `C` channel in `basis.npy`.

Once the preprocessing is complete, you can run CellART using the following script:

```python
import cellart
from pathlib import Path
import wandb
import os

# Preprocessed data
save_dir = "{YOUR_PREVIOUS_SAVE_DIR}"
# Directory to store all results
log_dir = "/{YOUR_RESULT_LOG_PATH}/cellart_result/"

manager = cellart.ExperimentManager(
    # Basic input data settings (must be specified)
    gene_map=os.path.join(save_dir, "gene_map.npy"),
    nuclei_mask=os.path.join(save_dir, "segmentation_mask.npy"),
    basis=os.path.join(save_dir, "basis.npy"),
    gene_names=os.path.join(save_dir, "filtered_gene_names.txt"),
    celltype_names=os.path.join(save_dir, "celltype_names.txt"),
    log_dir=log_dir,

    # Training parameters (adjust based on convergence and wandb visualization)
    epoch=200, 
    seg_training_epochs=10,
    deconv_warmup_epochs=100,

    pred_period=50,
    gpu="0"
)

# Update options
opt = manager.get_opt()
print(opt)

# Set up wandb for logging and visualization
run = wandb.init(project="CellART", dir=manager.get_log_dir(), config=opt,
                 name=os.path.basename(os.path.normpath(manager.get_log_dir())))

# Set up dataset
dataset = cellart.SSTDataset(manager)
gene_map_shape = dataset.gene_map.shape

# Initialize and train the CellART model
model = cellart.CellARTModel(manager, gene_map_shape, len(dataset.coords_starts))
model.train_model(dataset)
```

---

### How to Select a Proper Epoch Number?

The appropriate epoch number for training the CellART model may vary depending on the complexity of the dataset. Below are general guidelines for selecting the epoch parameters:

1. **Large and Complex Datasets**:
   For datasets with diverse cell types or complex gene expression patterns (like more than 3K genes selected), use higher epoch numbers to ensure sufficient training:
   ```python
   epoch = 400
   seg_training_epochs = 15
   deconv_warmup_epochs = 200
   ```
   Use the `wandb` package to monitor the training process, visualize the annotation results, and evaluate whether they meet expectations. If convergence is achieved earlier than expected, you can terminate training prematurely and check the results.

2. **Simpler Datasets**:
   For datasets with simpler cell type compositions (e.g., fewer subtypes, fewer gene number), the model typically converges faster. In these cases, you can use lower epoch numbers:
   ```python
   epoch = 200
   seg_training_epochs = 10
   deconv_warmup_epochs = 100
   ```


---

### Output

After training, the results will be stored in the `log_dir` directory. Key output files include:

- **epoch_{PRED_EPOCH}/cell_deconv.h5ad**: An [Annotated data](https://anndata.readthedocs.io/en/stable/) file containing segmentation and cell type annotation results.
- **new_segmentation_mask.npy**: A numpy file with the updated segmentation mask, where cell IDs correspond to the `obs_names` in the `cell_deconv.h5ad` file.
