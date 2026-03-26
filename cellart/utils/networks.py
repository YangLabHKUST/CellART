import torch
import numpy as np


# ============================================================
# Likelihood loss functions
# ============================================================
# All functions share the same interface:
#   count_matrix [N, G]: observed counts
#   log_lam [N, G]:      log(beta @ basis) + alpha + gamma
#   library_size [N, 1]: total UMI per spot
# Additional parameters are distribution-specific.

def poisson_loss(count_matrix, log_lam, library_size):
    """Poisson negative log-likelihood (original CellART)."""
    lam = torch.exp(log_lam)
    loss = -torch.mean(torch.sum(
        count_matrix * (torch.log(library_size + 1e-6) + log_lam) - library_size * lam,
        dim=1
    ))
    return loss


def nb_loss(count_matrix, log_lam, library_size, log_r):
    """Negative Binomial negative log-likelihood.
    log_r [G]: log of per-gene dispersion. r = exp(log_r).
    Large r -> Poisson; small r -> overdispersed.
    Variance = mu + mu^2 / r.
    """
    mu = library_size * torch.exp(log_lam)
    r = torch.exp(log_r).unsqueeze(0)
    log_prob = (
        torch.lgamma(count_matrix + r)
        - torch.lgamma(r)
        + r * torch.log(r + 1e-8)
        + count_matrix * torch.log(mu + 1e-8)
        - (r + count_matrix) * torch.log(r + mu + 1e-8)
    )
    return -torch.mean(torch.sum(log_prob, dim=1))


def zip_loss(count_matrix, log_lam, library_size, logit_pi):
    """Zero-Inflated Poisson negative log-likelihood.
    logit_pi [G]: logit of per-gene zero-inflation probability.
    pi = sigmoid(logit_pi). When pi = 0, reduces to Poisson.
    """
    mu = library_size * torch.exp(log_lam)
    pi = torch.sigmoid(logit_pi).unsqueeze(0)
    log_pi = torch.log(pi + 1e-8)
    log_1mpi = torch.log(1 - pi + 1e-8)
    # x = 0: log(pi + (1-pi)*exp(-mu)), via log-sum-exp for stability
    log_prob_zero = torch.logaddexp(log_pi, log_1mpi - mu)
    # x > 0: log(1-pi) + x*log(mu) - mu
    log_prob_pos = log_1mpi + count_matrix * torch.log(mu + 1e-8) - mu
    is_zero = (count_matrix < 0.5).float()
    log_prob = is_zero * log_prob_zero + (1 - is_zero) * log_prob_pos
    return -torch.mean(torch.sum(log_prob, dim=1))


def zinb_loss(count_matrix, log_lam, library_size, log_r, logit_pi):
    """Zero-Inflated Negative Binomial negative log-likelihood.
    log_r [G]: log of per-gene dispersion.
    logit_pi [G]: logit of per-gene zero-inflation probability.
    Reduces to NB when pi=0, to ZIP when r->inf, to Poisson when both.
    """
    mu = library_size * torch.exp(log_lam)
    r = torch.exp(log_r).unsqueeze(0)
    pi = torch.sigmoid(logit_pi).unsqueeze(0)
    log_1mpi = torch.log(1 - pi + 1e-8)
    # NB log-prob (used for x > 0)
    nb_log_prob = (
        torch.lgamma(count_matrix + r)
        - torch.lgamma(r)
        + r * torch.log(r + 1e-8)
        + count_matrix * torch.log(mu + 1e-8)
        - (r + count_matrix) * torch.log(r + mu + 1e-8)
    )
    # x = 0: log(pi + (1-pi) * (r/(r+mu))^r)
    nb_log_zero = r * (torch.log(r + 1e-8) - torch.log(r + mu + 1e-8))
    log_pi = torch.log(pi + 1e-8)
    log_prob_zero = torch.logaddexp(log_pi, log_1mpi + nb_log_zero)
    # x > 0
    log_prob_pos = log_1mpi + nb_log_prob
    is_zero = (count_matrix < 0.5).float()
    log_prob = is_zero * log_prob_zero + (1 - is_zero) * log_prob_pos
    return -torch.mean(torch.sum(log_prob, dim=1))


def tweedie_loss(count_matrix, log_lam, library_size, rho):
    """Tweedie deviance loss.
    rho [G] or scalar: p = 1 + sigmoid(rho), so p in (1, 2).
    p -> 1: Poisson-like; p -> 2: Gamma-like.
    Variance = phi * mu^p.
    """
    mu = library_size * torch.exp(log_lam)
    mu = torch.clamp(mu, min=1e-6)
    p = 1.0 + torch.sigmoid(rho)
    if p.dim() > 0:
        p = p.unsqueeze(0)
    log_mu = torch.log(mu)
    term1 = count_matrix * torch.exp((1 - p) * log_mu) / (1 - p)
    term2 = torch.exp((2 - p) * log_mu) / (2 - p)
    log_lik = term1 - term2
    return -torch.mean(torch.sum(log_lik, dim=1))


def _compute_loss(likelihood, count_matrix, log_lam, library_size, log_r=None, logit_pi=None, rho=None):
    """Dispatch to the selected loss function."""
    if likelihood == "poisson":
        return poisson_loss(count_matrix, log_lam, library_size)
    elif likelihood == "nb":
        return nb_loss(count_matrix, log_lam, library_size, log_r)
    elif likelihood == "zip":
        return zip_loss(count_matrix, log_lam, library_size, logit_pi)
    elif likelihood == "zinb":
        return zinb_loss(count_matrix, log_lam, library_size, log_r, logit_pi)
    elif likelihood == "tweedie":
        return tweedie_loss(count_matrix, log_lam, library_size, rho)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")


def _init_likelihood_params(module, likelihood, gene_num):
    """Add likelihood-specific learnable parameters to a module."""
    module.likelihood = likelihood
    if likelihood in ("nb", "zinb"):
        module.log_r = torch.nn.Parameter(torch.zeros(gene_num))
    if likelihood in ("zip", "zinb"):
        module.logit_pi = torch.nn.Parameter(torch.full((gene_num,), -2.0))
    if likelihood == "tweedie":
        module.rho = torch.nn.Parameter(torch.zeros(gene_num))


def _get_loss(module, count_matrix, log_lam, library_size):
    """Compute loss using the module's likelihood and parameters."""
    return _compute_loss(
        module.likelihood, count_matrix, log_lam, library_size,
        log_r=getattr(module, 'log_r', None),
        logit_pi=getattr(module, 'logit_pi', None),
        rho=getattr(module, 'rho', None),
    )


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
    def __init__(self, gene_num, hidden_dims, n_celltypes, patch_num, patch_emb_dim=16, likelihood="poisson"):
        super(DeconvNetPerSpotPatchEffect, self).__init__()
        self.hidden_dims = hidden_dims
        self.deconv_alpha_layer = DenseLayer(hidden_dims + patch_emb_dim, 1, zero_init=True)

        self.deconv_beta_layer = DenseLayer(hidden_dims, n_celltypes, zero_init=True)
        self.gamma = torch.nn.Parameter(torch.Tensor(patch_num, gene_num).zero_())
        self.patch_emb = torch.nn.Embedding(patch_num, patch_emb_dim)

        _init_likelihood_params(self, likelihood, gene_num)

    def forward(self, z, count_matrix, library_size, basis, patch_index):
        beta, alpha = self.deconv(z, patch_index)
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[patch_index].unsqueeze(0)
        return _get_loss(self, count_matrix, log_lam, library_size)

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
    def __init__(self, gene_num, hidden_dims, n_celltypes, patch_num, patch_emb_dim=16, likelihood="poisson"):
        super(DeconvNetNoPatchEffect, self).__init__()
        self.hidden_dims = hidden_dims
        self.deconv_alpha_layer = DenseLayer(hidden_dims, 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(hidden_dims, n_celltypes, zero_init=True)
        self.gamma = torch.nn.Parameter(torch.Tensor(1, gene_num).zero_())

        _init_likelihood_params(self, likelihood, gene_num)

    def forward(self, z, count_matrix, library_size, basis, patch_index):
        beta, alpha = self.deconv(z, patch_index)
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma
        return _get_loss(self, count_matrix, log_lam, library_size)

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