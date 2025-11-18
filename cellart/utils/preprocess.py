import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import bin2cell as b2c
import cv2
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
from .io import save_list, save_array
from scipy.sparse import coo_matrix

from tqdm import tqdm
import multiprocessing as mp
class SingleCellPreprocessor:
    def __init__(self, adata, celltype_col, save_path, st_gene_list = None, celltype_ref = None, sample_col = None):
        self.adata = adata
        self.celltype_ref = celltype_ref
        self.celltype_col = celltype_col
        self.sample_col = sample_col
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.st_gene_list = st_gene_list if st_gene_list is not None else adata.var_names

    def preprocess(self, hvg_method = None, n_hvg_group = 200, n_hvg = 2000):
        adata_ref = self.adata.copy()
        adata_ref.var_names_make_unique()
        # Remove mt-genes
        adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
                                 & np.array(~adata_ref.var_names.str.startswith("mt-"))
                                 & np.array(~adata_ref.var_names.str.startswith("MT-"))]
        if self.celltype_ref is not None:
            if not isinstance(self.celltype_ref, list):
                raise ValueError("'celltype_ref' must be a list!")
            else:
                adata_ref = adata_ref[[(t in self.celltype_ref) for t in adata_ref.obs[self.celltype_col].values.astype(str)],:]
        else:
            celltype_counts = adata_ref.obs[self.celltype_col].value_counts()
            celltype_ref = list(celltype_counts.index[celltype_counts > 1])
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[self.celltype_col].values.astype(str)], :]

        # Remove cells and genes with 0 counts
        sc.pp.filter_cells(adata_ref, min_genes=1)
        sc.pp.filter_genes(adata_ref, min_cells=1)

        # Take the subset of genes
        select_index = []
        for g in self.st_gene_list:
            if g in adata_ref.var_names:
                select_index.append(list(adata_ref.var_names).index(g))

        adata_ref = adata_ref[:, select_index]

        if hvg_method == "t-test":
            print("Select highly variable genes using t-test method...")
            # Select hvg
            adata_ref_log = adata_ref.copy()
            sc.pp.log1p(adata_ref_log)
            hvgs = self.select_hvgs(adata_ref_log, n_hvg_group)
            print("%d highly variable genes selected." % len(hvgs))
            adata_ref = adata_ref[:, hvgs]
        elif hvg_method == "seurat_v3":
            print("Select highly variable genes using Seurat v3 method...")
            sc.pp.highly_variable_genes(adata_ref, flavor="seurat_v3", n_top_genes=n_hvg)
            hvg = list(adata_ref.var.loc[adata_ref.var['highly_variable']==1].index)
            adata_ref = adata_ref[:, hvg]

            print("%d highly variable genes selected." % len(adata_ref.var.highly_variable))
        elif hvg_method == "combined":
            print("Select highly variable genes using combined method...")
            # Select hvg
            adata_ref_log = adata_ref.copy()
            sc.pp.log1p(adata_ref_log)
            hvgs_ttest = self.select_hvgs(adata_ref_log, n_hvg_group)

            sc.pp.highly_variable_genes(adata_ref, flavor="seurat_v3", n_top_genes=n_hvg)
            hvgs_seurat = list(adata_ref.var.loc[adata_ref.var['highly_variable']==1].index)
            hvgs = list(set(hvgs_ttest) | set(hvgs_seurat))
            adata_ref = adata_ref[:, hvgs]
            print("%d highly variable genes selected." % len(hvgs))
        elif hvg_method is None:
            print("No highly variable genes selected. All genes are used.")
        else:
            raise ValueError("Unknown hvg_method!")

        print("Calculate basis for deconvolution...")
        sc.pp.filter_cells(adata_ref, min_genes=1)
        sc.pp.normalize_total(adata_ref, target_sum=1)
        celltype_list = list(sorted(set(adata_ref.obs[self.celltype_col].values.astype(str))))
        basis = np.zeros((len(celltype_list), len(adata_ref.var.index)))

        if self.sample_col is not None:
            sample_list = list(sorted(set(adata_ref.obs[self.sample_col].values.astype(str))))
            for i in range(len(celltype_list)):
                c = celltype_list[i]
                tmp_list = []
                for j in range(len(sample_list)):
                    s = sample_list[j]
                    tmp = adata_ref[(adata_ref.obs[self.celltype_col].values.astype(str) == c) &
                            (adata_ref.obs[self.sample_col].values.astype(str) == s), :].X
                    if scipy.sparse.issparse(tmp):
                        tmp = tmp.toarray()
                    if tmp.shape[0] >= 3:
                        tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
                tmp_mean = np.mean(tmp_list, axis=0)
                if scipy.sparse.issparse(tmp_mean):
                    tmp_mean = tmp_mean.toarray()
                print("%d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
                basis[i, :] = tmp_mean
        else:
            for i in range(len(celltype_list)):
                c = celltype_list[i]
                tmp = adata_ref[adata_ref.obs[self.celltype_col].values.astype(str) == c, :].X
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                basis[i, :] = np.mean(tmp, axis=0).reshape((-1))

        # Save
        save_array(basis, self.save_path + "/basis.npy")
        save_list(celltype_list, self.save_path + "/celltype_names.txt")
        save_list(list(adata_ref.var_names), self.save_path + "/filtered_gene_names.txt")

    def select_hvgs(self, adata_ref_log, n_hvg_group):
        sc.tl.rank_genes_groups(adata_ref_log, groupby=self.celltype_col, method="t-test", key_added="ttest", use_raw=False)
        markers_df = pd.DataFrame(adata_ref_log.uns['ttest']['names']).iloc[0:n_hvg_group, :]
        genes = sorted(list(np.unique(markers_df.melt().value.values)))
        return genes


class VisiumHDPreprocessor:
    def __init__(self, bin_dir, source_he_image_path, spaceranger_image_path, save_path):
        self.adata = b2c.read_visium(bin_dir,
                                     source_image_path = source_he_image_path,
                                        spaceranger_image_path = spaceranger_image_path)
        self.adata.var_names_make_unique()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        # Save st gene list
        save_list(self.adata.var_names, self.save_path + "/st_gene_list.txt")

    def get_nuclei_segmentation(self, mpp = 0.5):
        b2c.scaled_he_image(self.adata, mpp=mpp, save_path = os.path.join(self.save_path, "scaled_he_image.tiff"))

        b2c.stardist(image_path=os.path.join(self.save_path, "scaled_he_image.tiff"),
                     labels_npz_path=os.path.join(self.save_path, "stardist_segmentation.npz"),
                     stardist_model="2D_versatile_he",
                     prob_thresh=0.01
                     )

        b2c.insert_labels(self.adata,
                          labels_npz_path=os.path.join(self.save_path, "stardist_segmentation.npz"),
                          basis="spatial",
                          spatial_key="spatial_cropped_150_buffer",
                          mpp=mpp,
                          labels_key="cell_id",
                          )

    def prepare_sst(self, filtered_gene_list, cell_col = "cell_id", cores = 20):
        adata_st = self.adata.copy()
        adata_st.var_names_make_unique()
        adata_st = adata_st[:, filtered_gene_list]

        map_width = adata_st.obs['array_col'].max() + 1
        map_height = adata_st.obs['array_row'].max() + 1

        adata_st.obs_names_make_unique()
        # Index Mapping
        adata_st.obs["spot_id"] = np.arange(adata_st.shape[0]) + 1
        spot_id_map = coo_matrix((adata_st.obs['spot_id'], (adata_st.obs['array_row'], adata_st.obs['array_col'])),
                                 shape=(map_height, map_width)).toarray()
        np.save(os.path.join(self.save_path, "spot_id_map.npy"), spot_id_map)
        # Save
        adata_st.write(self.save_path + "/filtered_adata_st.h5ad")

        # Turn in to adata
        n_processes = cores
        gene_names_chunks = np.array_split(filtered_gene_list, n_processes)
        processes = []

        # Create temp dir
        temp_dir = os.path.join(self.save_path, "per_gene_map")
        os.makedirs(temp_dir, exist_ok=True)
        for i, gene_chunk in enumerate(gene_names_chunks):
            adata_sub = adata_st[:, gene_chunk].copy()
            p = mp.Process(target=self.process_gene_chunk, args= (adata_sub, temp_dir, map_height, map_width))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Combine channel-wise
        gene_map = None
        for i, gene in tqdm(enumerate(filtered_gene_list), total=len(filtered_gene_list)):
            map = np.load(os.path.join(temp_dir, gene + ".npy"))
            if gene_map is None:
                gene_map = np.zeros((map.shape[0], map.shape[1], len(filtered_gene_list)), dtype=np.uint8)
                gene_map[:, :, i] = map
            else:
                gene_map[:, :, i] = map

        # Save the combined map
        save_array(gene_map, os.path.join(self.save_path, "gene_map.npy"))

        # Collect segmentation label
        df_seg = adata_st.obs[['array_row', 'array_col', cell_col]]
        seg_map = coo_matrix((df_seg[cell_col], (df_seg['array_row'], df_seg['array_col'])), shape=(map_height, map_width)).toarray()
        save_array(seg_map, os.path.join(self.save_path, "segmentation_mask.npy"))


    def process_gene_chunk(self, adata_sub, temp_dir, map_height, map_width):
        for gene in list(adata_sub.var_names):
            temp_df = adata_sub.obs[['array_row', 'array_col']]
            if scipy.sparse.issparse(adata_sub.X):
                temp_df[gene] = adata_sub[:, gene].X.toarray().flatten()
            else:
                temp_df[gene] = adata_sub[:, gene].X.flatten()
            map = coo_matrix((temp_df[gene], (temp_df['array_row'], temp_df['array_col'])), shape=(map_height, map_width)).toarray()
            save_array(map, os.path.join(temp_dir, gene + ".npy"))
            print(f"Gene {gene} processed.")

class XeniumPreprocessor:
    def __init__(self, transcripts_file, nucleus_boundary_10X, save_path, min_qv = 20):
        if transcripts_file.endswith(".parquet"):
            df = pd.read_parquet(transcripts_file)
        elif transcripts_file.endswith(".csv"):
            df = pd.read_csv(transcripts_file)
        else:
            raise ValueError("Unknown file format!")
        df['feature_name'] = df['feature_name'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        df = df[(df["qv"] >= min_qv) &
                (~df["feature_name"].str.startswith("NegControlProbe_")) &
                (~df["feature_name"].str.startswith("antisense_")) &
                (~df["feature_name"].str.startswith("NegControlCodeword_")) &
                (~df["feature_name"].str.startswith("BLANK_")) &
                (~df["feature_name"].str.startswith("UnassignedCodeword_"))]
        df.reset_index(inplace=True, drop=True)
        os.makedirs(save_path, exist_ok=True)

        self.nucleus_boundary_10X = nucleus_boundary_10X
        self.save_path = save_path
        df.to_csv(os.path.join(self.save_path, 'transcripts_filtered.csv'))
        self.df = df
        gene_names = df['feature_name'].unique()
        print('%d unique genes' % len(gene_names))
        self.gene_names = sorted(gene_names)
        save_list(gene_names, os.path.join(self.save_path, 'st_gene_list.txt'))

        self.map_width = int(np.ceil(df['x_location'].max())) + 1
        self.map_height = int(np.ceil(df['y_location'].max())) + 1


    # Multi-process of getting nuclei segmentation
    def get_nuclei_segmentation(self, cores = 30):
        nucleus_boundaries = pd.read_parquet(self.nucleus_boundary_10X)
        # All cell_id
        cell_ids = nucleus_boundaries['cell_id'].unique()
        n_processes = cores
        cell_id_chunks = np.array_split(cell_ids, n_processes)
        processes = []
        for i, cell_id_chunk in enumerate(cell_id_chunks):
            begin_mask_id = i * len(cell_id_chunk)
            p = mp.Process(target=self.process_nuclei_segmentation, args=(cell_id_chunk, nucleus_boundaries, begin_mask_id, i))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # Combine
        nuclei_mask_whole = np.zeros((self.map_height, self.map_width))
        for i in range(n_processes):
            current_mask = np.load(os.path.join(self.save_path, f"segmentation_mask_{i}.npy"))
            nuclei_mask_whole[current_mask > 0] = current_mask[current_mask > 0]

        # Save
        save_array(nuclei_mask_whole.astype(np.uint32), os.path.join(self.save_path, "segmentation_mask.npy"))
        # Delete temp files
        for i in range(n_processes):
            os.remove(os.path.join(self.save_path, f"segmentation_mask_{i}.npy"))

    def process_nuclei_segmentation(self, cell_id_chunk, nucleus_boundaries, begin_mask_id, process_id):
        nuclei_mask_whole = np.zeros((self.map_height, self.map_width))
        bar = tqdm(enumerate(cell_id_chunk), total=len(cell_id_chunk))
        for m, id in bar:
            mask_id = m + 1 + begin_mask_id
            single_nuclei_boundary = nucleus_boundaries[nucleus_boundaries['cell_id'] == id]
            single_nuclei_boundary = single_nuclei_boundary.reset_index(drop=True)
            if len(single_nuclei_boundary) > 0:
                # Detect how many nuclei in one cell
                df_nb = single_nuclei_boundary.copy()
                # Find none unique coordinates and its index
                df_nb = df_nb[df_nb.duplicated(subset=['vertex_x', 'vertex_y'], keep=False)]
                poly_num = int(len(df_nb) / 2)

                nuclei_poly_list = []
                for i in range(poly_num):
                    poly_index = (df_nb.index[i * 2], df_nb.index[i * 2 + 1])
                    df_poly = single_nuclei_boundary.loc[poly_index[0]:poly_index[1]]
                    x = df_poly.vertex_x
                    y = df_poly.vertex_y
                    polygons = [list(zip(x, y))]
                    polygons = np.array(polygons, 'int32')
                    nuclei_poly_list.append(polygons)
                cv2.fillPoly(nuclei_mask_whole, nuclei_poly_list, mask_id)
        # Save
        save_array(nuclei_mask_whole.astype(np.uint32), os.path.join(self.save_path, f"segmentation_mask_{process_id}.npy"))

    def prepare_sst(self, filtered_gene_list, cores = 20):
        # Prepare all patches
        n_processes = cores
        gene_names_chunks = np.array_split(filtered_gene_list, n_processes)
        processes = []

        # Create temp dir
        temp_dir = os.path.join(self.save_path, "per_gene_map")
        os.makedirs(temp_dir, exist_ok=True)
        for i, gene_chunk in enumerate(gene_names_chunks):
            df_sub = self.df[self.df['feature_name'].isin(gene_chunk)]
            p = mp.Process(target=self.process_gene_chunk, args=(gene_chunk, df_sub, temp_dir, self.map_width, self.map_height))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Combine channel-wise
        gene_map = None
        for i, gene in tqdm(enumerate(filtered_gene_list), total=len(filtered_gene_list)):
            map = np.load(os.path.join(temp_dir, gene + ".npy"))
            if gene_map is None:
                gene_map = np.zeros((map.shape[0], map.shape[1], len(filtered_gene_list)), dtype=np.uint8)
                gene_map[:, :, i] = map
            else:
                gene_map[:, :, i] = map

        # Save the combined map
        save_array(gene_map, os.path.join(self.save_path, "gene_map.npy"))

    def process_gene_chunk(self, gene_chunk, df_sub, temp_dir, map_width, map_height):
        for gene in gene_chunk:
            temp_df = df_sub[df_sub['feature_name'] == gene]
            temp_df['x'] = np.round(temp_df['x_location']).astype(int)
            temp_df['y'] = np.round(temp_df['y_location']).astype(int)
            # Add column count
            temp_df['MIDCount'] = 1
            temp_df = temp_df.groupby(['x', 'y']).sum().reset_index()
            map = coo_matrix((temp_df['MIDCount'], (temp_df['y'], temp_df['x'])), shape=(map_height, map_width)).toarray()
            save_array(map, os.path.join(temp_dir, gene + ".npy"))
            print(f"Gene {gene} processed.")