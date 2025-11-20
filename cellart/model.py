import os

import torch
from scipy.ndimage import center_of_mass
import cv2
import anndata
import distinctipy
import pandas as pd
import shutil
import wandb
import einops
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from .fpn.factory import make_spot_fpn_resnet
from .utils.networks import *
from .utils.operate import aggregate_gene_expression, split_mask_chunk, get_cell_size, get_cell_location, get_softmask_fast
import numpy as np
from .utils.io import load_list
from torch import nn
import torch.multiprocessing


class CellARTModel(nn.Module):
    def __init__(self, manager, gene_map_shape, patch_num):
        super(CellARTModel, self).__init__()
        self.manager = manager
        self.opt = manager.get_opt()
        self.logger = manager.get_logger()
        self.patch_num = patch_num
        # CPU Training is not supported
        self.device = torch.device("cuda")
        self.mask_size = (gene_map_shape[0], gene_map_shape[1])
        self.gene_num = gene_map_shape[2]
        self.gene_names = load_list(self.opt.gene_names)
        self.cell_chunk_size = self.opt.cell_chunk_size
        if not self.opt.nmf:
            self.celltype_names = load_list(self.opt.celltype_names)
            self.basis = np.load(self.opt.basis)
            self.basis = torch.from_numpy(self.basis).to(torch.float32).to(self.device)
            self.logger.info(f"Basis shape: {self.basis.shape}")
        else:
            # celltype_names: 1, 2, ..., factor_num
            self.celltype_names = [str(i) for i in range(self.opt.factor_num)]
            self.basis = torch.zeros(self.opt.factor_num, self.gene_num, device=self.device)
            self.logger.info(f"NMF basis shape: {self.basis.shape}")

        self._build_network()
        self._get_optimizer()

        self.new_segmentation_mask = None
        
        torch.multiprocessing.set_sharing_strategy('file_system')

    def _build_network(self, patch_emb_dim=16):
        model_fpn = make_spot_fpn_resnet(out_size=(self.opt.patch_size, self.opt.patch_size),
                                         in_channels=self.gene_num, num_classes=self.opt.deconv_emb_dim)
        self.backbone = model_fpn[0].to(self.device)
        self.fpn = model_fpn[1:].to(self.device)

        if not self.opt.nmf:
            print("=======> Cellsegmentation + CellAnnotation")
            self.deconv = DeconvNetPerSpotPatchEffect(gene_num=self.gene_num, hidden_dims=self.opt.deconv_emb_dim,
                                n_celltypes=self.basis.shape[0], patch_num=self.patch_num).to(self.device)
            if self.opt.no_patch_effect:
                print("=======> No Patch Effect")
                self.deconv = DeconvNetNoPatchEffect(gene_num=self.gene_num, hidden_dims=self.opt.deconv_emb_dim,
                                n_celltypes=self.basis.shape[0], patch_num=self.patch_num).to(self.device)
        else:
            print("=======> Cellsegmentation + NMF")
            self.deconv = NMFNet(gene_num=self.gene_num, hidden_dims=self.opt.deconv_emb_dim,
                                n_celltypes=self.basis.shape[0], patch_num=self.patch_num).to(self.device)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.opt.deconv_emb_dim + patch_emb_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.gene_num)
        ).to(self.device)
        self.slice_emb = torch.nn.Embedding(self.patch_num, patch_emb_dim).to(self.device)

    def _get_optimizer(self):
        self.optimizer_bb = torch.optim.Adam(self.backbone.parameters(), lr=self.opt.lr_bb)
        self.optimizer_fpn = torch.optim.Adam(self.fpn.parameters(), lr=self.opt.lr_fpn)
        self.optimizer_deconv = torch.optim.Adam(self.deconv.parameters(), lr=self.opt.lr_deconv)
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr_decoder)
        self.optimizer_slice_emb = torch.optim.Adam(self.slice_emb.parameters(), lr=self.opt.lr_decoder)
        self.optimizers = [self.optimizer_bb, self.optimizer_fpn, self.optimizer_deconv, self.optimizer_decoder, self.optimizer_slice_emb]

    def _get_batch(self, batch):
        expr, nucl, coords_h1, coords_w1 = batch
        expr, nucl = (
            expr[0].to(self.device).float(), nucl[0].to(self.device).long())
        return expr, nucl, coords_h1, coords_w1

    def _filter_border(self, nucl):
        border_nucl = torch.zeros_like(nucl)
        border_nucl[:, 0] = nucl[:, 0]
        border_nucl[:, -1] = nucl[:, -1]
        border_nucl[0, :] = nucl[0, :]
        border_nucl[-1, :] = nucl[-1, :]
        # Cell id of border nuclei excluding 0
        border_nucl_id = torch.unique(border_nucl)[1:]
        # Find the index of border nuclei in nucl
        border_nucl_idx = torch.where(torch.isin(nucl, border_nucl_id))
        # Set to zero
        nucl[border_nucl_idx] = 0

        return nucl, border_nucl_id

    def _forward_feature_map(self, expr):
        # CAUTION: The input expr is not normalized, using inplace operation to avoid memory issue
        expr /= expr.sum(dim=0, keepdim=True) + 1e-6
        expr *= self.opt.normalized_total
        expr = torch.log1p(expr)
        backbone_out = self.backbone(expr.unsqueeze(0))
        feature_map = self.fpn(backbone_out)[0]
        return feature_map

    def train_model(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=True, num_workers=self.opt.num_workers)
        deconv_loss_sum, decoder_loss_sum, total_spots = 0, 0, 0
        for epoch in range(1, self.opt.epochs + 1):
            for i, batch in enumerate(train_loader):
                self.train()
                self.backbone.apply(deactivate_batchnorm)

                expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
                patch_index = train_dataset.get_coords_index(coords_h1[0], coords_w1[0])

                # Filter border nuclei
                nucl, _ = self._filter_border(nucl)
                # Aggregate gene expression
                cell_gene = aggregate_gene_expression(nucl, einops.rearrange(expr, 'c h w -> h w c'), method='sum')
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                # Forward
                feature_map = self._forward_feature_map(expr)

                # Aggregate feature map
                cell_feature = aggregate_gene_expression(nucl, einops.rearrange(feature_map, 'c h w -> h w c'), method='mean')

                cell_umi = cell_gene.sum(dim=-1)
                cell_area = get_cell_size(nucl)
                # Filtered out small nuclei cell
                cell_feature = cell_feature[cell_area > self.opt.min_area]
                cell_gene = cell_gene[cell_area > self.opt.min_area]
                cell_umi = cell_umi[cell_area > self.opt.min_area]
                # Filtered out cell with small UMI
                cell_feature = cell_feature[cell_umi > self.opt.min_umi]
                cell_gene = cell_gene[cell_umi > self.opt.min_umi]
                cell_umi = cell_umi[cell_umi > self.opt.min_umi]

                if cell_feature.shape[0] == 0:
                    continue

                loss_deconv= self.deconv(cell_feature, cell_gene, cell_umi.unsqueeze(-1), self.basis, patch_index)
                if epoch > 1 and loss_deconv > 5e3:
                    self.logger.info(f"Epoch {epoch}, iter {i}, Skip patch {patch_index} due to some cell loss is too large: {loss_deconv}")
                    continue

                # Reconstruction
                patch_index = torch.tensor(patch_index).unsqueeze(0).to(cell_feature.device)
                patch_index = patch_index.repeat(cell_feature.shape[0])
                patch_emb = self.slice_emb(patch_index)
                z = self.decoder(torch.cat([cell_feature, patch_emb], dim=1))
                loss_decoder = torch.nn.functional.mse_loss(z, cell_gene) * self.opt.recon_loss_weight

                loss = loss_deconv + loss_decoder
                decoder_loss_sum += loss_decoder.item() * z.shape[0]
                deconv_loss_sum += loss_deconv.item() * z.shape[0]
                total_spots += z.shape[0]

                loss.backward()
                # Clip gradient
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.opt.gradient_clip)

                for optimizer in self.optimizers:
                    optimizer.step()

                self.logger.info(
                    f'Epoch {epoch}/{self.opt.epochs} | iter {i} | deconv_loss {loss_deconv.item()} | decoder_loss {loss_decoder.item()} | #cell {z.shape[0]}/{cell_area.shape[0]}')
                del expr, nucl, cell_gene, cell_feature, cell_umi, cell_area, feature_map, z, loss, loss_deconv, loss_decoder
                torch.cuda.empty_cache()


            if epoch % self.opt.save_period == 0:
                self.logger.info(f"Epoch {epoch}, saving model...")
                self.save(self.manager.get_checkpoint_dir() + '/model_epoch_{}.pth'.format(epoch))

            if epoch % self.opt.pred_period == 0:
                self.logger.info(f"Epoch {epoch}, predicting...")
                self.predict(train_dataset, save_dir=self.manager.get_log_dir() + '/epoch_{}'.format(epoch))
                adata = anndata.read_h5ad(
                    f'{self.manager.get_log_dir()}/epoch_{epoch}/cell_deconv.h5ad')
                argmax_deconv_map = np.argmax(adata.obsm['deconv_beta'], axis=1)
                # Project into celltype
                celltype_names = adata.uns['celltype_names']
                ct = [celltype_names[i] for i in argmax_deconv_map]
                adata.obs['celltype'] = ct
                colors = distinctipy.get_colors(len(celltype_names) + 1, pastel_factor=0.1, rng=15)
                fig, ax = plt.subplots(1, 1, figsize=(int(adata.obs['y'].max()/adata.obs['x'].max()*10), 10))
                for i in range(len(colors)):
                    # (0,0) is on the top left corner
                    if i == 0:
                        continue
                    x = adata.obs['x'][adata.obs['celltype'] == celltype_names[i - 1]]
                    y = adata.obs['y'][adata.obs['celltype'] == celltype_names[i - 1]]
                    ax.scatter(y, x, color=colors[i], s=0.3,
                               label=celltype_names[i - 1])
                ax.invert_yaxis()
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                fig.tight_layout()
                wandb.log({"Annotation": wandb.Image(fig)}, commit=False)

            if epoch == self.opt.deconv_warmup_epochs:
                self.training_predict_segmentation(train_dataset)
                train_dataset.setting_new_segmentation(self.new_segmentation_mask)
                print("======> Finish cell segmentation")

            wandb.log({"deconv_loss": deconv_loss_sum / total_spots,
                       "decoder_loss": decoder_loss_sum / (self.opt.recon_loss_weight + 1e-6) / total_spots})
            deconv_loss_sum, decoder_loss_sum, total_spots = 0, 0, 0

    def predict(self, test_dataset, save_dir=None):
        self.backbone.apply(deactivate_batchnorm)
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=self.opt.num_workers)

        # Make a temp dir
        # If save_pixel_feature is True, save_dir should be provided
        assert save_dir is not None or not self.opt.save_pixel_feature, "save_dir should be provided if save_pixel_feature is True"

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if self.opt.save_pixel_feature:
                os.makedirs(save_dir + '/pixel_feature', exist_ok=True)

        os.makedirs(self.manager.get_log_dir() + '/temp', exist_ok=True)


        with torch.no_grad():
            all_border_nucl_id = []
            for i, batch in enumerate(test_loader):
                print(f"Off Predicting {i}/{len(test_loader)}")
                expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
                nucl, border_nucl_id = self._filter_border(nucl)
                all_border_nucl_id.append(border_nucl_id)
                patch_index = test_dataset.get_coords_index(coords_h1[0], coords_w1[0])

                cell_gene = aggregate_gene_expression(nucl, einops.rearrange(expr, 'c h w -> h w c'), method='sum')
                # Forward
                feature_map = self._forward_feature_map(expr)

                if self.opt.save_pixel_feature:
                    np.save(save_dir + f'/pixel_feature/off_{coords_h1[0]}_{coords_w1[0]}_feature.npy', feature_map.cpu().numpy().transpose(1, 2, 0))

                cell_feature = aggregate_gene_expression(nucl, einops.rearrange(feature_map, 'c h w -> h w c'),
                                                         method='mean')
                if torch.sum(nucl) == 0:
                    continue

                cell_location = get_cell_location(nucl)
                cell_location = cell_location + torch.tensor([coords_h1, coords_w1]).to(cell_location.device)
                cell_ids = torch.unique(nucl)[1:]

                beta, _ = self.deconv.deconv(cell_feature, patch_index)
                # Save in temp, with prefix "off_"
                np.save(self.manager.get_log_dir() + '/temp/off_{}_cell_gene.npy'.format(i), cell_gene.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/off_{}_cell_feature.npy'.format(i), cell_feature.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/off_{}_cell_location.npy'.format(i), cell_location.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/off_{}_cell_ids.npy'.format(i), cell_ids.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/off_{}_cell_beta.npy'.format(i), beta.cpu().numpy())


                del expr, nucl, cell_gene, cell_feature, cell_location, cell_ids, beta, feature_map
                torch.cuda.empty_cache()

            all_border_nucl_id = torch.cat(all_border_nucl_id, dim=0)
            all_border_nucl_id = set(all_border_nucl_id.cpu().numpy())

            # Deal with boundary of the patches
            test_dataset.set_shifting_state("vertical")
            for i, batch in enumerate(test_loader):
                print(f"Vertical Predicting {i}/{len(test_loader)}")
                expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
                nucl, border_nucl_id = self._filter_border(nucl)
                patch_index = test_dataset.get_coords_index(coords_h1[0], coords_w1[0])

                # Only preserve cell with border nuclei
                all_border_nucl_id_torch = torch.tensor(list(all_border_nucl_id)).to(nucl.device)
                nucl = torch.where(torch.isin(nucl, all_border_nucl_id_torch), nucl, torch.zeros_like(nucl))
                # Delete the border nuclei in all_border_nucl_id
                all_border_nucl_id = all_border_nucl_id - set(torch.unique(nucl).cpu().numpy())

                cell_gene = aggregate_gene_expression(nucl, einops.rearrange(expr, 'c h w -> h w c'), method='sum')

                # Forward
                feature_map = self._forward_feature_map(expr)

                if self.opt.save_pixel_feature:
                    np.save(save_dir + f'/pixel_feature/vertical_{coords_h1[0]}_{coords_w1[0]}_feature.npy', feature_map.cpu().numpy().transpose(1, 2, 0))

                cell_feature = aggregate_gene_expression(nucl, einops.rearrange(feature_map, 'c h w -> h w c'),
                                                         method='mean')

                if torch.sum(nucl) == 0:
                    continue

                cell_location = get_cell_location(nucl)
                cell_location = cell_location + torch.tensor([coords_h1, coords_w1]).to(cell_location.device)
                cell_ids = torch.unique(nucl)[1:]

                beta, _ = self.deconv.deconv(cell_feature, patch_index)
                # Save in temp, with prefix "vertical_"
                np.save(self.manager.get_log_dir() + '/temp/vertical_{}_cell_gene.npy'.format(i), cell_gene.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/vertical_{}_cell_feature.npy'.format(i), cell_feature.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/vertical_{}_cell_location.npy'.format(i), cell_location.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/vertical_{}_cell_ids.npy'.format(i), cell_ids.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/vertical_{}_cell_beta.npy'.format(i), beta.cpu().numpy())


                del expr, nucl, cell_gene, cell_feature, cell_location, cell_ids, beta, feature_map
                torch.cuda.empty_cache()

            test_dataset.set_shifting_state("horizontal")
            for i, batch in enumerate(test_loader):
                print(f"Horizontal Predicting {i}/{len(test_loader)}")
                expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
                nucl, border_nucl_id = self._filter_border(nucl)
                patch_index = test_dataset.get_coords_index(coords_h1[0], coords_w1[0])

                # Only preserve cell with border nuclei
                all_border_nucl_id_torch = torch.tensor(list(all_border_nucl_id)).to(nucl.device)
                nucl = torch.where(torch.isin(nucl, all_border_nucl_id_torch), nucl, torch.zeros_like(nucl))
                all_border_nucl_id = all_border_nucl_id - set(torch.unique(nucl).cpu().numpy())

                cell_gene = aggregate_gene_expression(nucl, einops.rearrange(expr, 'c h w -> h w c'), method='sum')

                # Forward
                feature_map = self._forward_feature_map(expr)

                if self.opt.save_pixel_feature:
                    np.save(save_dir + f'/pixel_feature/horizontal_{coords_h1[0]}_{coords_w1[0]}_feature.npy',
                        feature_map.cpu().numpy().transpose(1, 2, 0))

                cell_feature = aggregate_gene_expression(nucl, einops.rearrange(feature_map, 'c h w -> h w c'),
                                                         method='mean')
                if torch.sum(nucl) == 0:
                    continue

                cell_location = get_cell_location(nucl)
                cell_location = cell_location + torch.tensor([coords_h1, coords_w1]).to(cell_location.device)
                cell_ids = torch.unique(nucl)[1:]

                beta, _ = self.deconv.deconv(cell_feature, patch_index)
                # Save in temp, with prefix "horizontal_"
                np.save(self.manager.get_log_dir() + '/temp/horizontal_{}_cell_gene.npy'.format(i), cell_gene.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/horizontal_{}_cell_feature.npy'.format(i), cell_feature.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/horizontal_{}_cell_location.npy'.format(i), cell_location.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/horizontal_{}_cell_ids.npy'.format(i), cell_ids.cpu().numpy())
                np.save(self.manager.get_log_dir() + '/temp/horizontal_{}_cell_beta.npy'.format(i), beta.cpu().numpy())

                del expr, nucl, cell_gene, cell_feature, cell_location, cell_ids, beta, feature_map
                torch.cuda.empty_cache()

            test_dataset.set_shifting_state("off")

        # Concat all in the temp dir
        for prefix in ["off_", "vertical_", "horizontal_"]:
            cell_gene_list = []
            cell_feature_list = []
            cell_location_list = []
            cell_ids_list = []
            cell_beta_list = []
            for i in range(len(test_loader)):
                if os.path.exists(self.manager.get_log_dir() + f'/temp/{prefix}{i}_cell_gene.npy'):
                    cell_gene_list.append(np.load(self.manager.get_log_dir() + f'/temp/{prefix}{i}_cell_gene.npy'))
                    cell_feature_list.append(np.load(self.manager.get_log_dir() + f'/temp/{prefix}{i}_cell_feature.npy'))
                    cell_location_list.append(np.load(self.manager.get_log_dir() + f'/temp/{prefix}{i}_cell_location.npy'))
                    cell_ids_list.append(np.load(self.manager.get_log_dir() + f'/temp/{prefix}{i}_cell_ids.npy'))
                    cell_beta_list.append(np.load(self.manager.get_log_dir() + f'/temp/{prefix}{i}_cell_beta.npy'))

            cell_gene = np.concatenate(cell_gene_list, axis=0)
            cell_feature = np.concatenate(cell_feature_list, axis=0)
            cell_location = np.concatenate(cell_location_list, axis=0)
            cell_ids = np.concatenate(cell_ids_list, axis=0)
            cell_beta = np.concatenate(cell_beta_list, axis=0)

            if prefix == "off_":
                cell_gene_off = cell_gene
                cell_feature_off = cell_feature
                cell_location_off = cell_location
                cell_ids_off = cell_ids
                cell_beta_off = cell_beta
            elif prefix == "vertical_":
                cell_gene_vertical = cell_gene
                cell_feature_vertical = cell_feature
                cell_location_vertical = cell_location
                cell_ids_vertical = cell_ids
                cell_beta_vertical = cell_beta
            else:
                cell_gene_horizontal = cell_gene
                cell_feature_horizontal = cell_feature
                cell_location_horizontal = cell_location
                cell_ids_horizontal = cell_ids
                cell_beta_horizontal = cell_beta

        # Concatenate all
        cell_gene = np.concatenate([cell_gene_off, cell_gene_vertical, cell_gene_horizontal], axis=0)
        cell_feature = np.concatenate([cell_feature_off, cell_feature_vertical, cell_feature_horizontal], axis=0)
        cell_location = np.concatenate([cell_location_off, cell_location_vertical, cell_location_horizontal], axis=0)
        cell_ids = np.concatenate([cell_ids_off, cell_ids_vertical, cell_ids_horizontal], axis=0)
        cell_beta = np.concatenate([cell_beta_off, cell_beta_vertical, cell_beta_horizontal], axis=0)

        gene_names = self.gene_names
        if save_dir is None:
            save_dir = self.manager.get_log_dir()
        else:
            os.makedirs(save_dir, exist_ok=True)

        # Save Anndata
        # Give each spot a unique id
        df_obs = pd.DataFrame(index=cell_ids)
        df_obs['x'] = cell_location[:, 0]
        df_obs['y'] = cell_location[:, 1]

        adata = anndata.AnnData(X = cell_gene, obs = df_obs, var = pd.DataFrame(index = gene_names))
        adata.obsm['latent'] = cell_feature
        adata.uns['celltype_names'] = self.celltype_names
        adata.obsm['deconv_beta'] = cell_beta
        adata.obs["cell_id"] = adata.obs_names.astype(float).astype(int).astype(str)
        adata.obs.set_index("cell_id", inplace=True)
        # Since there is some overlap of the bundary patch, we need to remove the duplicated cell id
        # (last patch: shape[1]-patch_size, but last second patch is from shape[1]-patch_size to shape[1])
        adata = adata[~adata.obs.index.duplicated(keep='first')]
        temp = pd.DataFrame(adata.obsm["deconv_beta"], columns=adata.uns["celltype_names"],
                            index=adata.obs.index)
        adata.obs["celltype"] = temp.idxmax(axis=1)

        # if nmf: save basis
        if self.opt.nmf:
            adata.uns['basis'] = self.deconv.get_basis().cpu().numpy()
        else:
            adata.uns['basis'] = self.basis.cpu().numpy()

        adata.write_h5ad(save_dir + '/cell_deconv.h5ad')

        # Collect feature map
        if self.opt.save_pixel_feature:
            print("======> Collecting feature map")
            whole_feature_map = np.zeros((self.mask_size[0], self.mask_size[1], self.opt.deconv_emb_dim))
            feature_map_dir = save_dir + '/pixel_feature'
            # off_{coords_h1[0]}_{coords_w1[0]}_feature.npy
            for f in os.listdir(feature_map_dir):
                if f.endswith('.npy') and f.startswith('off_'):
                    feature_map = np.load(os.path.join(feature_map_dir, f))
                    coords_h1, coords_w1 = map(int, f.split('_')[1:3])
                    whole_feature_map[coords_h1:coords_h1 + self.opt.patch_size,
                                     coords_w1:coords_w1 + self.opt.patch_size] = feature_map

            for f in os.listdir(feature_map_dir):
                if f.endswith('.npy') and f.startswith('vertical_'):
                    feature_map = np.load(os.path.join(feature_map_dir, f))
                    coords_h1, coords_w1 = map(int, f.split('_')[1:3])
                    if coords_h1 + self.opt.patch_size > self.mask_size[0] or coords_w1 + self.opt.patch_size > self.mask_size[1]:
                        continue
                    low_h1 = coords_h1 + self.opt.patch_size - self.opt.patch_size // 4 - self.opt.patch_size // 8
                    high_h1 = coords_h1 + self.opt.patch_size - self.opt.patch_size // 4 + self.opt.patch_size // 8
                    whole_feature_map[low_h1:high_h1, coords_w1:coords_w1 + self.opt.patch_size] = feature_map[
                        (self.opt.patch_size - self.opt.patch_size // 4 - self.opt.patch_size // 8):(self.opt.patch_size - self.opt.patch_size // 4 + self.opt.patch_size // 8), :]

            for f in os.listdir(feature_map_dir):
                if f.endswith('.npy') and f.startswith('horizontal_'):
                    feature_map = np.load(os.path.join(feature_map_dir, f))
                    coords_h1, coords_w1 = map(int, f.split('_')[1:3])
                    if coords_h1 + self.opt.patch_size > self.mask_size[0] or coords_w1 + self.opt.patch_size > self.mask_size[1]:
                        continue
                    low_w1 = coords_w1 + self.opt.patch_size - self.opt.patch_size // 4 - self.opt.patch_size // 8
                    high_w1 = coords_w1 + self.opt.patch_size - self.opt.patch_size // 4 + self.opt.patch_size // 8
                    whole_feature_map[coords_h1:coords_h1 + self.opt.patch_size, low_w1:high_w1] = feature_map[:,
                        (self.opt.patch_size - self.opt.patch_size // 4 - self.opt.patch_size // 8):(self.opt.patch_size - self.opt.patch_size // 4 + self.opt.patch_size // 8)]

            # Save feature map
            np.save(save_dir + '/feature_map.npy', whole_feature_map)

            # Delete feature map dir
            shutil.rmtree(feature_map_dir)

        # Delete temp dir
        shutil.rmtree(self.manager.get_log_dir() + '/temp')

    def training_predict_segmentation(self, dataset):
        cell_chunk_size = self.cell_chunk_size
        self.eval()
        self.backbone.apply(deactivate_batchnorm)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=self.opt.num_workers)
        cell_predictor = CellPredictNet().cuda()
        optimizer = torch.optim.Adam(cell_predictor.parameters(), lr=1e-3,
                                     betas=(0.9,
                                            0.999),
                                     weight_decay=0.0001)
        for e in range(self.opt.seg_training_epochs):
            for ite, batch in enumerate(train_loader):
                expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
                nucl, border_nucl_id = self._filter_border(nucl)
                # Forward
                with torch.no_grad():
                    feature_map = self._forward_feature_map(expr)
                cell_ids = torch.unique(nucl)[1:]
                if len(cell_ids) == 0:
                    continue
                cell_chunks = []
                for i in range(0, len(cell_ids), cell_chunk_size):
                    if min(i + cell_chunk_size, len(cell_ids)) - i < cell_chunk_size // 2:
                        continue

                    cell_chunks.append((i, min(i + cell_chunk_size, len(cell_ids))))
                if len(cell_chunks) == 0:
                    continue

                minipatch_size = 48
                half_size = minipatch_size // 2

                feature_map_pad = torch.nn.functional.pad(feature_map, (half_size, half_size, half_size, half_size),
                                                          mode='constant', value=0).detach()
                for cell_chunk in cell_chunks:
                    binary_masks_chunk = split_mask_chunk(nucl, cell_chunk[0], cell_chunk[1])
                    N, H, W = binary_masks_chunk.shape
                    centers = []
                    binary_masks_pad = torch.nn.functional.pad(binary_masks_chunk,
                                                               (half_size, half_size, half_size, half_size),
                                                               mode='constant',
                                                               value=0)
                    for i in range(N):
                        cy, cx = center_of_mass(binary_masks_pad[i].cpu().numpy())  # Returns float coordinates
                        centers.append([int(cy), int(cx)])
                    centers = torch.tensor(centers, device=feature_map_pad.device)
                    mini_feat = torch.zeros((N, feature_map_pad.shape[0], minipatch_size, minipatch_size),
                                            device=feature_map_pad.device)
                    mini_nucl = torch.zeros((N, minipatch_size, minipatch_size), device=feature_map_pad.device)
                    for i in range(N):
                        cy, cx = centers[i]
                        cy = int(cy)
                        cx = int(cx)
                        mini_feat[i] = feature_map_pad[:, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                        mini_nucl[i] = binary_masks_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.opt.dilation_kernel_size, self.opt.dilation_kernel_size))
                    soft_masks = np.zeros((len(mini_nucl), mini_nucl[0].shape[0], mini_nucl[0].shape[1]))
                    dilated_masks = np.zeros((len(mini_nucl), mini_nucl[0].shape[0], mini_nucl[0].shape[1]))
                    for c in range(len(mini_nucl)):
                        cell_nuclei_mask = mini_nucl[c]
                        cell_nuclei_mask = cell_nuclei_mask.cpu().numpy()
                        I = get_softmask_fast(cell_nuclei_mask, tau=self.opt.tau)
                        soft_masks[c] = I
                        dilated_masks[c] = cv2.dilate(cell_nuclei_mask.astype('uint8'), kernel, iterations=self.opt.dilation_iter_num)

                    pred = cell_predictor(torch.from_numpy(soft_masks).float().cuda().unsqueeze(1), mini_feat)
                    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1., 1.]).cuda())
                    loss_nucl = criterion_ce(pred, mini_nucl.long())
                    loss_mask = mini_nucl + torch.from_numpy(1 - dilated_masks).int().cuda()
                    loss_nucl = loss_nucl * loss_mask

                    loss = torch.sum(loss_nucl) / torch.sum(loss_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.logger.info(
                        f"Epoch: {e}, Iter: {ite}, Cell chunk: {cell_chunk}, Loss_nucl: {torch.sum(loss_nucl) / torch.sum(loss_mask)}")

        self.logger.info("=======> Training done, Start predicting new cell segmentation mask")
        all_border_nucl_id = []
        new_segmentation_mask_dict = {}
        for ite, batch in enumerate(train_loader):
            print(f"Iter {ite}/{len(train_loader)} in no shift")
            expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
            nucl, border_nucl_id = self._filter_border(nucl)
            all_border_nucl_id.append(border_nucl_id)
            # Forward
            with torch.no_grad():
                feature_map = self._forward_feature_map(expr)

            cell_ids = torch.unique(nucl)[1:]
            if len(cell_ids) == 0:
                continue
            cell_chunks = []

            for i in range(0, len(cell_ids), cell_chunk_size):
                cell_chunks.append((i, min(i + cell_chunk_size, len(cell_ids))))

            minipatch_size = 48
            half_size = minipatch_size // 2
            feature_map_pad = torch.nn.functional.pad(feature_map, (half_size, half_size, half_size, half_size),
                                                      mode='constant', value=0).detach()
            all_center = []
            all_pred = []

            for cell_chunk in cell_chunks:
                binary_masks_chunk = split_mask_chunk(nucl, cell_chunk[0], cell_chunk[1])
                N, H, W = binary_masks_chunk.shape
                centers = []
                binary_masks_pad = torch.nn.functional.pad(binary_masks_chunk,
                                                           (half_size, half_size, half_size, half_size),
                                                           mode='constant', value=0)
                for i in range(N):
                    cy, cx = center_of_mass(binary_masks_pad[i].cpu().numpy())  # Returns float coordinates
                    centers.append([int(cy), int(cx)])
                centers = torch.tensor(centers, device=feature_map_pad.device)
                mini_feat = torch.zeros((N, feature_map_pad.shape[0], minipatch_size, minipatch_size),
                                        device=feature_map_pad.device)
                mini_nucl = torch.zeros((N, minipatch_size, minipatch_size), device=feature_map_pad.device)
                for i in range(N):
                    cy, cx = centers[i]
                    cy = int(cy)
                    cx = int(cx)
                    mini_feat[i] = feature_map_pad[:, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                    mini_nucl[i] = binary_masks_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                soft_masks = np.zeros((len(mini_nucl), mini_nucl[0].shape[0], mini_nucl[0].shape[1]))
                for c in range(len(mini_nucl)):
                    cell_nuclei_mask = mini_nucl[c]
                    cell_nuclei_mask = cell_nuclei_mask.cpu().numpy()
                    I = get_softmask_fast(cell_nuclei_mask)
                    soft_masks[c] = I
                pred = cell_predictor(torch.from_numpy(soft_masks).float().cuda().unsqueeze(1), mini_feat)
                pred_prob = torch.nn.functional.softmax(pred, dim=1)[:, 1]

                all_center.append(centers.cpu().numpy())
                all_pred.append(pred_prob.detach().cpu().numpy())

            # Concatenate all the chunks
            all_center = np.concatenate(all_center)
            all_pred = np.concatenate(all_pred)
            pred_probs_pad = np.zeros((len(all_center), H + 2 * half_size, W + 2 * half_size))
            for i in range(len(all_center)):
                cy, cx = all_center[i]
                cy = int(cy)
                cx = int(cx)
                pred_probs_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size] = all_pred[i]
            pred_probs = pred_probs_pad[:, half_size:-half_size, half_size:-half_size]
            bgd_probs = 1 - pred_probs.max(axis=0)
            # Concat cell and background
            cell_probs = np.concatenate([bgd_probs[np.newaxis], pred_probs], axis=0)
            final_seg = np.argmax(cell_probs, axis=0)
            # Make prediciton projection to the original nucl index
            dictionary = dict(
                zip(np.arange(0, len(cell_ids) + 1), torch.cat([torch.tensor([0]), cell_ids.cpu()]).numpy()))

            final_seg = np.vectorize(dictionary.get)(final_seg)
            final_seg = np.where(nucl.cpu().numpy() > 0, nucl.cpu().numpy(), final_seg)
            # If the predicted cell touch the border, set to 0 and add to the border_nucl_id
            final_seg, border_nucl_id = self._filter_border(torch.tensor(final_seg).to(nucl.device))
            all_border_nucl_id.append(border_nucl_id)
            new_segmentation_mask_dict[(coords_h1, coords_w1)] = final_seg.cpu().numpy()

            # Delete
            del expr, nucl, feature_map

        all_border_nucl_id = torch.cat(all_border_nucl_id, dim=0)
        all_border_nucl_id = set(all_border_nucl_id.cpu().numpy())
        dataset.set_shifting_state("vertical")
        vertical_segmentation_mask_dict = {}
        for ite, batch in enumerate(train_loader):
            print(f"Iter {ite}/{len(train_loader)} in vertical shift")
            expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
            nucl, border_nucl_id = self._filter_border(nucl)
            # Only preserve cell with border nuclei
            all_border_nucl_id_torch = torch.tensor(list(all_border_nucl_id)).to(nucl.device)
            nucl = torch.where(torch.isin(nucl, all_border_nucl_id_torch), nucl, torch.zeros_like(nucl))
            all_border_nucl_id = all_border_nucl_id - set(torch.unique(nucl).cpu().numpy())
            # Forward
            with torch.no_grad():
                feature_map = self._forward_feature_map(expr)

            cell_ids = torch.unique(nucl)[1:]
            if len(cell_ids) == 0:
                continue
            # Divide the cell into chunks
            cell_chunks = []
            for i in range(0, len(cell_ids), cell_chunk_size):
                cell_chunks.append((i, min(i + cell_chunk_size, len(cell_ids))))

            minipatch_size = 48
            half_size = minipatch_size // 2

            feature_map_pad = torch.nn.functional.pad(feature_map, (half_size, half_size, half_size, half_size),
                                                        mode='constant', value=0).detach()
            all_center = []
            all_pred = []
            for cell_chunk in cell_chunks:
                binary_masks_chunk = split_mask_chunk(nucl, cell_chunk[0], cell_chunk[1])
                N, H, W = binary_masks_chunk[cell_chunk[0]:cell_chunk[1]].shape
                centers = []
                binary_masks_pad = torch.nn.functional.pad(binary_masks_chunk[cell_chunk[0]:cell_chunk[1]],
                                                           (half_size, half_size, half_size, half_size),
                                                           mode='constant', value=0)
                for i in range(N):
                    cy, cx = center_of_mass(binary_masks_pad[i].cpu().numpy())  # Returns float coordinates
                    centers.append([int(cy), int(cx)])
                centers = torch.tensor(centers, device=feature_map_pad.device)
                mini_feat = torch.zeros((N, feature_map_pad.shape[0], minipatch_size, minipatch_size),
                                        device=feature_map_pad.device)
                mini_nucl = torch.zeros((N, minipatch_size, minipatch_size), device=feature_map_pad.device)
                for i in range(N):
                    cy, cx = centers[i]
                    cy = int(cy)
                    cx = int(cx)
                    mini_feat[i] = feature_map_pad[:, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                    mini_nucl[i] = binary_masks_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                # print(f"N: {N}, H: {H}, W: {W}, binary_masks_chunk: {binary_masks_chunk.shape}, mini_nucl: {mini_nucl.shape}")
                try:
                    soft_masks = np.zeros((len(mini_nucl), mini_nucl[0].shape[0], mini_nucl[0].shape[1]))
                except:
                    # Zero cell in this chunk
                    continue
                for c in range(len(mini_nucl)):
                    cell_nuclei_mask = mini_nucl[c]
                    cell_nuclei_mask = cell_nuclei_mask.cpu().numpy()
                    I = get_softmask_fast(cell_nuclei_mask)
                    soft_masks[c] = I

                pred = cell_predictor(torch.from_numpy(soft_masks).float().cuda().unsqueeze(1), mini_feat)
                pred_prob = torch.nn.functional.softmax(pred, dim=1)[:, 1]

                all_center.append(centers.cpu().numpy())
                all_pred.append(pred_prob.detach().cpu().numpy())
            # Concatenate all the chunks
            all_center = np.concatenate(all_center)
            all_pred = np.concatenate(all_pred)
            pred_probs_pad = np.zeros((len(all_center), H + 2 * half_size, W + 2 * half_size))
            for i in range(len(all_center)):
                cy, cx = all_center[i]
                cy = int(cy)
                cx = int(cx)
                pred_probs_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size] = all_pred[i]
            pred_probs = pred_probs_pad[:, half_size:-half_size, half_size:-half_size]
            bgd_probs = 1 - pred_probs.max(axis=0)
            # Concat cell and background
            cell_probs = np.concatenate([bgd_probs[np.newaxis], pred_probs], axis=0)
            final_seg = np.argmax(cell_probs, axis=0)
            # Make prediciton projection to the original nucl index
            dictionary = dict(
                zip(np.arange(0, len(cell_ids) + 1), torch.cat([torch.tensor([0]), cell_ids.cpu()]).numpy()))

            final_seg = np.vectorize(dictionary.get)(final_seg)
            final_seg = np.where(nucl.cpu().numpy() > 0, nucl.cpu().numpy(), final_seg)
            # If the predicted cell touch the border, set to 0 and add to the border_nucl_id
            final_seg, border_nucl_id = self._filter_border(torch.tensor(final_seg).to(nucl.device))
            all_border_nucl_id = all_border_nucl_id.union(set(border_nucl_id))
            vertical_segmentation_mask_dict[(coords_h1, coords_w1)] = final_seg.cpu().numpy()

            del expr, nucl, feature_map

        dataset.set_shifting_state("horizontal")
        horizontal_segmentation_mask_dict = {}
        for ite, batch in enumerate(train_loader):
            print(f"Iter {ite}/{len(train_loader)} in horizontal shift")
            expr, nucl, coords_h1, coords_w1 = self._get_batch(batch)
            nucl, border_nucl_id = self._filter_border(nucl)

            # Only preserve cell with border nuclei
            all_border_nucl_id_torch = torch.tensor(list(all_border_nucl_id)).to(nucl.device)
            nucl = torch.where(torch.isin(nucl, all_border_nucl_id_torch), nucl, torch.zeros_like(nucl))
            all_border_nucl_id = all_border_nucl_id - set(torch.unique(nucl).cpu().numpy())

            # Forward
            with torch.no_grad():
                feature_map = self._forward_feature_map(expr)
            cell_ids = torch.unique(nucl)[1:]
            if len(cell_ids) == 0:
                continue
            cell_chunks = []

            for i in range(0, len(cell_ids), cell_chunk_size):
                cell_chunks.append((i, min(i + cell_chunk_size, len(cell_ids))))

            minipatch_size = 48
            half_size = minipatch_size // 2
            feature_map_pad = torch.nn.functional.pad(feature_map, (half_size, half_size, half_size, half_size),
                                                      mode='constant', value=0).detach()

            all_center = []
            all_pred = []

            for cell_chunk in cell_chunks:
                binary_masks_chunk = split_mask_chunk(nucl, cell_chunk[0], cell_chunk[1])
                N, H, W = binary_masks_chunk.shape
                centers = []
                binary_masks_pad = torch.nn.functional.pad(binary_masks_chunk,
                                                           (half_size, half_size, half_size, half_size),
                                                           mode='constant', value=0)
                for i in range(N):
                    cy, cx = center_of_mass(binary_masks_pad[i].cpu().numpy())  # Returns float coordinates
                    centers.append([int(cy), int(cx)])
                centers = torch.tensor(centers, device=feature_map_pad.device)
                mini_feat = torch.zeros((N, feature_map_pad.shape[0], minipatch_size, minipatch_size),
                                        device=feature_map_pad.device)
                mini_nucl = torch.zeros((N, minipatch_size, minipatch_size), device=feature_map_pad.device)
                for i in range(N):
                    cy, cx = centers[i]
                    cy = int(cy)
                    cx = int(cx)
                    mini_feat[i] = feature_map_pad[:, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                    mini_nucl[i] = binary_masks_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size]
                # print(
                #     f"N: {N}, H: {H}, W: {W}, binary_masks_chunk: {binary_masks_chunk.shape}, mini_nucl: {mini_nucl.shape}")
                soft_masks = np.zeros((len(mini_nucl), mini_nucl[0].shape[0], mini_nucl[0].shape[1]))
                for c in range(len(mini_nucl)):
                    cell_nuclei_mask = mini_nucl[c]
                    cell_nuclei_mask = cell_nuclei_mask.cpu().numpy()
                    I = get_softmask_fast(cell_nuclei_mask)
                    soft_masks[c] = I

                pred = cell_predictor(torch.from_numpy(soft_masks).float().cuda().unsqueeze(1), mini_feat)
                pred_prob = torch.nn.functional.softmax(pred, dim=1)[:, 1]

                all_center.append(centers.cpu().numpy())
                all_pred.append(pred_prob.detach().cpu().numpy())
            # Concatenate all the chunks
            all_center = np.concatenate(all_center)
            all_pred = np.concatenate(all_pred)
            pred_probs_pad = np.zeros((len(all_center), H + 2 * half_size, W + 2 * half_size))
            for i in range(len(all_center)):
                cy, cx = all_center[i]
                cy = int(cy)
                cx = int(cx)
                pred_probs_pad[i, cy - half_size:cy + half_size, cx - half_size:cx + half_size] = all_pred[i]

            pred_probs = pred_probs_pad[:, half_size:-half_size, half_size:-half_size]
            bgd_probs = 1 - pred_probs.max(axis=0)
            # Concat cell and background
            cell_probs = np.concatenate([bgd_probs[np.newaxis], pred_probs], axis=0)
            final_seg = np.argmax(cell_probs, axis=0)
            # Make prediciton projection to the original nucl index
            dictionary = dict(
                zip(np.arange(0, len(cell_ids) + 1), torch.cat([torch.tensor([0]), cell_ids.cpu()]).numpy()))
            final_seg = np.vectorize(dictionary.get)(final_seg)
            final_seg = np.where(nucl.cpu().numpy() > 0, nucl.cpu().numpy(), final_seg)
            horizontal_segmentation_mask_dict[(coords_h1, coords_w1)] = final_seg

            del expr, nucl, feature_map

        dataset.set_shifting_state("off")

        # Merge all the segmentation mask
        new_segmentation_mask = np.zeros(dataset.nuclei_mask.shape)

        for coords_h1, coords_w1 in vertical_segmentation_mask_dict.keys():
            # Only update the non-zero region
            current_region = new_segmentation_mask[coords_h1:coords_h1 + self.opt.patch_size,
                             coords_w1:coords_w1 + self.opt.patch_size]
            vertical_region = vertical_segmentation_mask_dict[(coords_h1, coords_w1)][0:current_region.shape[0],
                              0:current_region.shape[1]]
            x, y = np.where(vertical_region > 0)
            current_region[x, y] = vertical_region[x, y]
            new_segmentation_mask[coords_h1:coords_h1 + self.opt.patch_size,
            coords_w1:coords_w1 + self.opt.patch_size] = current_region

        for coords_h1, coords_w1 in horizontal_segmentation_mask_dict.keys():
            # Only update the non-zero region
            current_region = new_segmentation_mask[coords_h1:coords_h1 + self.opt.patch_size,
                             coords_w1:coords_w1 + self.opt.patch_size]
            horizontal_region = horizontal_segmentation_mask_dict[(coords_h1, coords_w1)][0:current_region.shape[0],
                                0:current_region.shape[1]]
            x, y = np.where(horizontal_region > 0)
            current_region[x, y] = horizontal_region[x, y]
            new_segmentation_mask[coords_h1:coords_h1 + self.opt.patch_size,
            coords_w1:coords_w1 + self.opt.patch_size] = current_region

        for coords_h1, coords_w1 in new_segmentation_mask_dict.keys():
            current_region = new_segmentation_mask[coords_h1:coords_h1 + self.opt.patch_size,
                             coords_w1:coords_w1 + self.opt.patch_size]
            region = new_segmentation_mask_dict[(coords_h1, coords_w1)]
            x, y = np.where(region > 0)
            current_region[x, y] = region[x, y]
            new_segmentation_mask[coords_h1:coords_h1 + self.opt.patch_size,
            coords_w1:coords_w1 + self.opt.patch_size] = current_region

        self.new_segmentation_mask = new_segmentation_mask
        # Save
        np.save(self.manager.get_log_dir() + '/new_segmentation_mask.npy', new_segmentation_mask)
        # wandb show the segmentation result
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(new_segmentation_mask > 0)
        wandb.log({"new_segmentation": wandb.Image(fig)}, commit=False)

    def save(self, path):
        """Save the model state
        """
        save_dict = {'model_state': self.state_dict()}
        torch.save(save_dict, path)

    def load(self, path, strict = True):
        """Load a model state from a checkpoint file
        """
        checkpoint_file = path
        checkpoints = torch.load(checkpoint_file)
        self.load_state_dict(checkpoints['model_state'], strict=strict)

























