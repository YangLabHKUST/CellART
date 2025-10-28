import torch
import numpy as np

class DenseLayer(torch.nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):
        super().__init__()
        self.linear = torch.nn.Linear(c_in, c_out)
        if zero_init:
            torch.nn.init.zeros_(self.linear.weight.data)
        else:
            torch.nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        torch.nn.init.zeros_(self.linear.bias.data)
    def forward(self,node_feats):
        node_feats = self.linear(node_feats)
        return node_feats

class DeconvNetPerSpotPatchEffect(torch.nn.Module):
    def __init__(self, gene_num, hidden_dims, n_celltypes, patch_num, patch_emb_dim = 16):
        super(DeconvNetPerSpotPatchEffect, self).__init__()
        self.hidden_dims = hidden_dims
        self.deconv_alpha_layer = DenseLayer(hidden_dims + patch_emb_dim, 1, zero_init=True)

        self.deconv_beta_layer = DenseLayer(hidden_dims, n_celltypes, zero_init=True)
        self.gamma = torch.nn.Parameter(torch.Tensor(patch_num, gene_num).zero_())
        self.patch_emb = torch.nn.Embedding(patch_num, patch_emb_dim)

    def forward(self, z, count_matrix, library_size, basis, patch_index):
        beta, alpha = self.deconv(z, patch_index)
        # Add entropy loss to encourage sparsity
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[patch_index].unsqueeze(0)
        lam = torch.exp(log_lam)

        decon_loss = -torch.mean(torch.sum(
                count_matrix * (torch.log(library_size + 1e-6) + log_lam) - library_size * lam, dim=1)
            )

        return decon_loss

    def deconv(self, z, patch_index):
        # patch_index: int -> N x 1 tensor
        patch_index = torch.tensor(patch_index).unsqueeze(0).to(z.device)
        patch_index = patch_index.repeat(z.shape[0])
        beta = self.deconv_beta_layer(torch.nn.functional.elu(z))
        beta = torch.nn.functional.softmax(beta, dim=1)
        patch_emb = self.patch_emb(patch_index)
        alpha = self.deconv_alpha_layer(torch.nn.functional.elu(torch.cat([z, patch_emb], dim=1)))
        return beta, alpha
    

class DeconvNetNoPatchEffect(torch.nn.Module):
    def __init__(self, gene_num, hidden_dims, n_celltypes, patch_num, patch_emb_dim = 16):
        super(DeconvNetNoPatchEffect, self).__init__()
        self.hidden_dims = hidden_dims
        self.deconv_alpha_layer = DenseLayer(hidden_dims, 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(hidden_dims, n_celltypes, zero_init=True)
        self.gamma = torch.nn.Parameter(torch.Tensor(1, gene_num).zero_())

    def forward(self, z, count_matrix, library_size, basis, patch_index):
        beta, alpha = self.deconv(z, patch_index)
        # Add entropy loss to encourage sparsity
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma
        lam = torch.exp(log_lam)

        decon_loss = -torch.mean(torch.sum(
                count_matrix * (torch.log(library_size + 1e-6) + log_lam) - library_size * lam, dim=1)
            )

        return decon_loss

    def deconv(self, z, patch_index):
        beta = self.deconv_beta_layer(torch.nn.functional.elu(z))
        beta = torch.nn.functional.softmax(beta, dim=1)
        alpha = self.deconv_alpha_layer(torch.nn.functional.elu(z))
        return beta, alpha


class NMFNet(torch.nn.Module):
    def __init__(self, gene_num, hidden_dims, n_celltypes, patch_num, patch_emb_dim = 16):
        super(NMFNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.deconv_alpha_layer = DenseLayer(hidden_dims + patch_emb_dim, 1, zero_init=True)

        self.deconv_beta_layer = DenseLayer(hidden_dims, n_celltypes, zero_init=True)
        self.facotors = torch.nn.Parameter(torch.Tensor(n_celltypes, gene_num).
                                           uniform_(-np.sqrt(6/(n_celltypes+gene_num)), np.sqrt(6/(n_celltypes+gene_num))))

        self.gamma = torch.nn.Parameter(torch.Tensor(patch_num, gene_num).zero_())
        self.patch_emb = torch.nn.Embedding(patch_num, patch_emb_dim)

    def get_basis(self):
        basis = torch.nn.functional.softmax(self.facotors, dim=1)
        return basis

    def forward(self, z, count_matrix, library_size, basis, patch_index):
        basis = self.get_basis()
        beta, alpha = self.deconv(z, patch_index)
        # Add entropy loss to encourage sparsity
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[patch_index].unsqueeze(0)
        lam = torch.exp(log_lam)

        decon_loss = -torch.mean(torch.sum(
                count_matrix * (torch.log(library_size + 1e-6) + log_lam) - library_size * lam, dim=1)
            )

        return decon_loss

    def deconv(self, z, patch_index):
        # patch_index: int -> N x 1 tensor
        patch_index = torch.tensor(patch_index).unsqueeze(0).to(z.device)
        patch_index = patch_index.repeat(z.shape[0])
        beta = self.deconv_beta_layer(torch.nn.functional.elu(z))
        beta = torch.nn.functional.softmax(beta, dim=1)
        patch_emb = self.patch_emb(patch_index)
        alpha = self.deconv_alpha_layer(torch.nn.functional.elu(torch.cat([z, patch_emb], dim=1)))
        return beta, alpha

class CellPredictNet(torch.nn.Module):
    def __init__(self, cond_channels = 128):
        super(CellPredictNet, self).__init__()
        self.cond_channel = cond_channels

        # FLiM
        self.FLiMmul = torch.nn.Sequential(
            torch.nn.Conv2d(self.cond_channel, 32, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )

        self.predict = torch.nn.Conv2d(1, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, nuclei_soft_mask, cond):
        f_mul = self.FLiMmul(cond)
        x = nuclei_soft_mask * f_mul
        x = self.predict(x)
        return x

def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()