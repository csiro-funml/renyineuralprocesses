"""Module for all the loss of Neural Process Family."""
import abc
import math
from addict import Dict
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

__all__ = ["CNPFLoss", "ELBOLossLNPF", "NLLLossLNPF",
           "GVIRenyiLoss", "TestLoss"]


def sum_from_nth_dim(t, dim):
    """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
    return t.view(*t.shape[:dim], -1).sum(-1) # returns (n_z, b)


def sum_log_prob(prob, sample):
    """Compute log probability then sum all but the z_samples and batch."""
    # size = [n_z_samples, batch_size, *]
    log_p = prob.log_prob(sample)
    # size = [n_z_samples, batch_size]
    sum_log_p = sum_from_nth_dim(log_p, 2)
    return sum_log_p


def grid_to_number_points(X):
    H, W, dim = X.shape[-3:]
    X = X.view(*X.shape[:-3], H * W, dim)
    return X


class BaseLossNPF(nn.Module, abc.ABC):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self, reduction="mean", is_force_mle_eval=True, alpha='Renyi_1.0'):
        super().__init__()
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval
        self.alpha = float(alpha.split('_')[1])

    def forward(self, pred_outputs, Y_trgt):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        p_yCc, z_samples, q_zCc, q_zCct = pred_outputs

        if self.training:
            loss = self.get_loss(p_yCc, z_samples, q_zCc, q_zCct, Y_trgt)
        else:
            # always uses NPML for evaluation
            if self.is_force_mle_eval:
                q_zCct = None
            loss = NLLLossLNPF.get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, Y_trgt)

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")

    @abc.abstractmethod
    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, Y_trgt):
        """Compute the Neural Process Loss

        Parameters
        ------
        p_yCc: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, z_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor, size=[1].
        """
        pass


class CNPFLoss(BaseLossNPF):
    """Losss for conditional neural process (suf-)family [1]."""

    def get_loss(self, p_yCc, _, q_zCc, ___, Y_trgt):
        assert q_zCc is None
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # size = [batch_size]
        nll = -sum_log_p_yCz.squeeze(0)
        return nll


class ELBOLossLNPF(BaseLossNPF):
    """Approximate conditional ELBO [1].

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, Y_trgt):
        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        kl_z = kl_divergence(q_zCct, q_zCc)
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        return -(E_z_sum_log_p_yCz - E_z_kl)


class NLLLossLNPF(BaseLossNPF):
    """
    Compute the approximate negative log likelihood for Neural Process family[?].

     Notes
    -----
    - might be high variance
    - biased
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    References
    ----------
    [?]
    """

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, Y_trgt):

        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # computes approximate LL in a numerically stable way
        # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
        # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
        # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
        # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
        # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)

        # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # uses importance sampling weights if necessary
        if q_zCct is not None:
            # All latents are treated as independent. size = [n_z_samples, batch_size]
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            # importance sampling : multiply \prod_t p(y^t|z)) by q(z|y_cntxt) / q(z|y_cntxt, y_trgt)
            # i.e. add log q(z|y_cntxt) - log q(z|y_cntxt, y_trgt)
            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
            if self.alpha != 1.0:
                sum_log_w_k = (1- self.alpha) * sum_log_w_k
        else:
            sum_log_w_k = sum_log_p_yCz

        # log_sum_exp_z ... . size = [batch_size]
        log_S_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)

        # - log(n_z_samples)
        log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)

        if self.alpha != 1.0 and q_zCct is not None:
            log_E_z_sum_p_yCz = log_E_z_sum_p_yCz/(1-self.alpha)

        # NEGATIVE log likelihood
        return -log_E_z_sum_p_yCz




class KLDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, q_zt, q_zc):
        if q_zt is None:
            return torch.tensor(0)
        kl_z = kl_divergence(q_zt, q_zc)

        # \sum_l ... . size = [batch_size]
        kl_d = sum_from_nth_dim(kl_z, 1)

        # self validation
        # tr_pq = torch.sum(q_zt.scale **2 / q_zc.scale **2, dim=-1)
        # mean_pq = torch.sum((q_zc.mean - q_zt.mean)**2 / q_zc.scale **2, dim=-1)
        # k = q_zc.mean.shape[-1]
        # log_pq = 2*torch.sum(torch.log(q_zc.scale) - torch.log(q_zt.scale), dim=-1)
        # kl_self = 0.5 * (tr_pq + mean_pq - k + log_pq)
        # print("error",torch.mean((kl_self - kl_d)**2).item())
        return kl_d



def to_numpy(x):
    return x.detach().cpu().numpy()

def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def tnp_loss(outputs, batch, num_samples, training=True, reduce_ll=True):
    # from attrdict import AttrDict
    from addict import Dict
    py, z, pz, qz = outputs
    outs = Dict()
    if training:
        if num_samples > 1:
            # K * B * N
            recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
            likelihood = to_numpy(recon)
            # print("likelihood", likelihood[0])
            # K * B
            log_qz = qz.log_prob(z).sum(-1)
            log_pz = pz.log_prob(z).sum(-1)
            prior = to_numpy(log_pz)
            poster = to_numpy(log_qz)
            # print("prior", prior)
            # print("poster", poster)

            # K * B
            log_w = recon.sum(-1) + log_pz - log_qz

            outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]
        else:
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(qz, pz).sum(-1).mean()
            outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

    else:
        if num_samples is None:
            ll = py.log_prob(batch.y).sum(-1)
        else:
            y = torch.stack([batch.y] * num_samples)
            if reduce_ll:
                ll = logmeanexp(py.log_prob(y).sum(-1))
            else:
                ll = py.log_prob(y).sum(-1)
        num_ctx = batch.xc.shape[-2]
        if reduce_ll:
            outs.ctx_ll = ll[..., :num_ctx].mean()
            outs.tar_ll = ll[..., num_ctx:].mean()
        else:
            outs.ctx_ll = ll[..., :num_ctx]
            outs.tar_ll = ll[..., num_ctx:]
    return outs

class RenyiDivergence:
    def __init__(self, alpha=1.0):
        # self.alpha = float(alpha.split('_')[1]) # only when it is a string
        self.alpha = alpha

    def __call__(self, outputs, batch, num_samples=1, training=True, reduce_ll=True):
        py, z, pz, qz = outputs
        if 'mask_xc' in batch.keys(): # gridded input for ConvNP
            batch.y = grid_to_number_points(batch.y)
        outs = Dict()
        batch_size = batch.x.shape[0]


        if training:
            # if num_samples > 1:
                # K * B * N
            recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
            if qz is not None:
                # K * B
                log_qz = qz.log_prob(z).reshape(num_samples, batch_size, -1).sum(-1)
                log_pz = pz.log_prob(z).reshape(num_samples, batch_size, -1).sum(-1)
                # K * B
                log_w = recon.sum(-1) + log_pz - log_qz
            else:
                log_w = recon.sum(-1)

            if self.alpha == 1.0: # KL
                outs.loss = -logmeanexp(log_w).mean() / batch.y.shape[-2]
            else: # alpha divergence (including mle)
                log_w_alpha = (1 - self.alpha) * log_w
                logmeanexp_alpha = logmeanexp(log_w_alpha)/(1-self.alpha)
                outs.loss = -logmeanexp_alpha.mean() / batch.y.shape[-2]

            # else:
            #     outs.recon = py.log_prob(batch.y).sum(-1).mean()
            #     outs.kld = kl_divergence(qz, pz).sum(-1).mean()
            #     outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

        else:
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            else:
                y = torch.stack([batch.y] * num_samples)
                if reduce_ll:
                    ll = logmeanexp(py.log_prob(y).sum(-1))
                else:
                    ll = py.log_prob(y).sum(-1)
            if 'mask_xc' not in batch.keys():
                num_ctx = batch.xc.shape[-2]
                if reduce_ll:
                    outs.ctx_ll = ll[..., :num_ctx].mean()
                    outs.tar_ll = ll[..., num_ctx:].mean()
                else:
                    outs.ctx_ll = ll[..., :num_ctx]
                    outs.tar_ll = ll[..., num_ctx:]
            else:
                outs.ctx_ll = ll[batch.mask_xc.int()==1].mean()
                outs.tar_ll = ll[batch.mask_xt.int()==1].mean()
        return outs

class LLBNP(RenyiDivergence):
    def __init__(self, alpha):
        super().__init__(alpha)

    def compute_ll(self, py, y, reduce_ll=True):
        ll = py.log_prob(y).sum(-1)
        if ll.dim() == 3 and reduce_ll:
            ll = logmeanexp(ll)
        return ll

    def __call__(self, outputs, batch, num_samples, training=True, reduce_ll=True):
        outs = Dict()
        if self.alpha == 0.0:
            py_base, py = outputs
            outs.ll_base = self.compute_ll(py_base, batch.y).mean()
            outs.ll = self.compute_ll(py, batch.y).mean()
            outs.loss = -outs.ll_base - outs.ll
        else:
            py_base, py, z, pz, qz = outputs
            recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
            recon_base = py_base.log_prob(stack(batch.y, num_samples)).sum(-1)
            recon = recon_base + recon
            # K * B
            log_qz = qz.log_prob(z).sum(-1)
            log_pz = pz.log_prob(z).sum(-1)

            # K * B
            log_w = recon.sum(-1) + log_pz - log_qz
            if self.alpha == 1.0:  # KL
                outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]

            else:  # alpha divergence:
                log_w_alpha = (1 - self.alpha) * log_w
                logmeanexp_alpha = logmeanexp(log_w_alpha) / (1 - self.alpha)
                outs.loss = -logmeanexp_alpha.mean() / batch.x.shape[-2]

        return outs
    

class OldRenyiDivergence:
    def __init__(self, alpha=0.75, approx=False):
        self.alpha = alpha
        self.approx = approx

    def __call__(self, outputs, targets, mask=None):

        y_dist, z, p, q = outputs

        q_likelihood = sum_log_prob(q, z)

        prior = sum_log_prob(p, z)

        n_z_samples = z.shape[0]
        if mask is not None:
            targets = grid_to_number_points(targets)
            likelihood = (y_dist.log_prob(stack(targets, n_z_samples))*mask[None, :, :, None]).sum(-1)
            n_target = torch.sum(mask, dim=-1).mean()
        else:
            likelihood = y_dist.log_prob(stack(targets, n_z_samples)).sum(-1)
            n_target = targets.shape[-2]

        ratio = prior + likelihood.sum(-1) - q_likelihood
        if self.alpha != 1:
            ratio = (1 - self.alpha) * ratio
            logmeanexp = (torch.logsumexp(ratio, 0) - math.log(n_z_samples)) / (1-self.alpha)
        else:  # KL
            # logmeanexp = torch.logsumexp(ratio, 0) - math.log(n_z_samples)
            logmeanexp = ratio
        loss = -logmeanexp.mean() / n_target
        return loss


class LocalRenyiDivergence:
    def __init__(self, alpha=0.75, approx=False):
        self.alpha = alpha
        self.approx = approx

    def __call__(self, outputs, targets, mask=None):

        y_dist, z, p, q = outputs

        q_likelihood = sum_log_prob(q, z)

        prior = sum_log_prob(p, z)

        n_z_samples = z.shape[0]
        if mask is not None:
            targets = grid_to_number_points(targets)
            likelihood = (y_dist.log_prob(stack(targets, n_z_samples))*mask[None, :, :, None]).sum(-1)
            n_target = torch.sum(mask, dim=-1).mean()
        else:
            likelihood = y_dist.log_prob(stack(targets, n_z_samples)).sum(-1)
            n_target = targets.shape[-2]

        if self.alpha != 1:
            ratio = (1 - self.alpha) * (prior - q_likelihood)
            logmeanexp = (torch.logsumexp(ratio, 0) - math.log(n_z_samples)) / (1-self.alpha) + likelihood.sum(-1)
        else:  # KL
            # logmeanexp = torch.logsumexp(ratio, 0) - math.log(n_z_samples)
            logmeanexp = prior - q_likelihood + likelihood.sum(-1)
        loss = -logmeanexp.mean() / n_target
        return loss

class AnalyticalRenyiDivergence(nn.Module):
    def __init__(self, alhpa=0.95):
        super().__init__()
        self.alpha = alhpa

    def forward(self, z, q_zt, q_zc, alpha=None):
        if q_zt is None:
            return torch.tensor(0)

        if alpha is not None:  # for testing purpose
            self.alpha = alpha

        # sanity check
        if len(q_zc.mean) == 2:
            q_zc_mean = q_zc.mean
            q_zc_scale = q_zc.stddev
            q_zt_mean = q_zt.mean
            q_zt_scale = q_zt.stddev
        else:
            bs = q_zc.mean.shape[0]
            q_zc_mean = q_zc.mean.reshape(bs, -1)
            q_zc_scale = q_zc.stddev.reshape(bs, -1)
            q_zt_mean = q_zt.mean.reshape(bs, -1)
            q_zt_scale = q_zt.stddev.reshape(bs, -1)
        assert len(q_zc_scale.shape) == 2, "q_z shape not equal to 2"

        sigma_star = self.alpha * q_zc_scale ** 2 + (1 - self.alpha) * q_zt_scale ** 2
        # tr_pq = torch.sum(q_zt.scale **2 / q_zc.scale **2, dim=-1)
        mean_pq = torch.sum((q_zc_mean - q_zt_mean) ** 2 / sigma_star, dim=-1)

        det_num = torch.sum(torch.log(sigma_star), dim=-1)
        det_denom = (1 - self.alpha) * torch.sum(2 * torch.log(q_zt_scale), dim=-1) + self.alpha * torch.sum(
            2 * torch.log(q_zc_scale), dim=-1)
        det_term = 1 / (self.alpha * (self.alpha - 1)) * (det_num - det_denom)

        alpha_renyi = 0.5 * (mean_pq - det_term)

        alpha_renyi = self.alpha * alpha_renyi
        # kl_z = kl_divergence(q_zt, q_zc)
        # kl_d = sum_from_nth_dim(kl_z, 1)

        # print("error",torch.mean((alpha_renyi - kl_d)**2).item())
        return alpha_renyi


class WassersteinDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, q_zt, q_zc):
        if q_zt is None:
            return torch.tensor(0)
        # sanity check
        if len(q_zc.mean) == 2:
            q_zc_mean = q_zc.mean
            q_zc_scale = q_zc.stddev
            q_zt_mean = q_zt.mean
            q_zt_scale = q_zt.stddev
        else:
            q_zc_mean = q_zc.mean.squeeze(1)
            q_zc_scale = q_zc.stddev.squeeze(1)
            q_zt_mean = q_zt.mean.squeeze(1)
            q_zt_scale = q_zt.stddev.squeeze(1)
        assert len(q_zc_scale.shape) == 2, "q_z shape not equal to 2"

        norm = torch.sum((q_zt_mean - q_zc_mean) ** 2, dim=-1)
        tr = torch.sum((q_zt_scale - q_zc_scale) ** 2, dim=-1)
        w_dist = norm + tr
        return w_dist


class NegativeLoglikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p_yCc, Y_trgt):
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        # Y_trgt shape: (bs, n_target, y_dim)
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        mean_log_p_yCz = sum_log_p_yCz / Y_trgt.shape[-2]  # average over the number of points

        # size = [batch_size]
        nll = -mean_log_p_yCz.squeeze(0)
        return nll


class MaxMeanDiscrepancy(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MaxMeanDiscrepancy, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss




class GVIRenyiLoss(nn.Module, abc.ABC):
    """
    Compute the GVI loss for members of the conditional neural process (sub-)family.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self,
                 divergence=None,
                 scorerule=None,
                 reduction="mean", is_force_mle_eval=True,
                 model_name=None):
        super().__init__()
        self.approx = True if model_name == 'ConvNP' else False
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval
        self.divergence = self.get_divergence(divergence)
        self.scorerule = self.get_scorerule(scorerule)

    def get_divergence(self, name):
        if name == 'KL':
            return KLDivergence()
        if 'Renyi' in name:
            if '_' not in name:
                return OldRenyiDivergence()
            else:
                alpha = float(name.split('_')[1])
                # return AnalyticalRenyiDivergence(alpha=alpha)
                return OldRenyiDivergence(alpha=alpha, approx=self.approx)
        if name == 'Wasserstein':
            return WassersteinDistance()

    def get_scorerule(self, name):
        if name == 'log':
            return NegativeLoglikelihood()

    def forward(self, pred_outputs, Y_trgt, mask=None):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        loss = self.divergence(pred_outputs, Y_trgt, mask=mask)
        return loss


class LocalGVILoss(nn.Module, abc.ABC):
    """
    Compute the GVI loss for members of the conditional neural process (sub-)family.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self,
                 divergence=None,
                 scorerule=None,
                 reduction="mean", is_force_mle_eval=True,
                 model_name=None):
        super().__init__()
        self.approx = True if model_name == 'ConvNP' else False
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval
        self.divergence = self.get_divergence(divergence)
        self.scorerule = self.get_scorerule(scorerule)

    def get_divergence(self, name):
        if name == 'KL':
            return KLDivergence()
        if 'Renyi' in name:
            if '_' not in name:
                return RenyiDivergence()
            else:
                alpha = float(name.split('_')[1])
                # return AnalyticalRenyiDivergence(alpha=alpha)
                return LocalRenyiDivergence(alpha=alpha, approx=self.approx)
        if name == 'Wasserstein':
            return WassersteinDistance()

    def get_scorerule(self, name):
        if name == 'log':
            return NegativeLoglikelihood()

    def forward(self, pred_outputs, Y_trgt, mask=None):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        loss = self.divergence(pred_outputs, Y_trgt, mask=mask)
        return loss


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


class TestLoss(nn.Module, abc.ABC):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self,
                 prob_metric='LL',
                 deter_metric='RMSE',
                 reduction="mean", is_force_mle_eval=True):
        super().__init__()
        self.reduction = reduction
        self.deter_metric = deter_metric
        self.prob_metric = prob_metric
        self.is_force_mle_eval = is_force_mle_eval
        self.prob = self.get_prob(prob_metric)
        self.deter = self.get_deter(deter_metric)

    def get_deter(self, name):
        if name == 'RMSE':
            return torch.nn.MSELoss(reduction='none')

    def get_prob(self, name):
        if name == 'LL':
            return NegativeLoglikelihood()

    def forward(self, pred_outputs, Y_trgt, X_cnxt, X_trgt=None, reduce_ll=True, mask=None, batch=None):
        def process_loss(loss):
            if self.reduction is None:
                # size = [batch_size]
                return loss
            elif self.reduction == "mean":
                # size = [1]
                return loss.mean(0)
            elif self.reduction == "sum":
                # size = [1]
                return loss.sum(0)
            else:
                raise ValueError(f"Unknown {self.reduction}")

        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """

        p_yt = pred_outputs[0]
        num_samples = p_yt.mean.shape[0] if len(p_yt.mean.shape) != len(Y_trgt.shape) else None

        if self.prob_metric == 'LL':
            if num_samples is None:
                ll = p_yt.log_prob(Y_trgt).sum(-1)
            else:
                if mask is None:
                    y = torch.stack([Y_trgt] * num_samples)
                    if reduce_ll:
                        ll = logmeanexp(p_yt.log_prob(y).sum(-1))
                    else:
                        ll = p_yt.log_prob(y).sum(-1)

                else:
                    y = torch.stack([Y_trgt] * num_samples)
                    ll = logmeanexp((p_yt.log_prob(y) * mask[None, :,:, None]).sum(-1))

            if mask is None:
                num_ctx = X_cnxt.shape[-2]
                if reduce_ll:
                    ll = ll[..., num_ctx:].mean()  # average of the target points
                else:
                    ll = ll[..., num_ctx:]
            else:
                temp = torch.sum(mask, dim=-1)
                # ll =  torch.sum(ll, dim=-1)/torch.sum(mask, dim=-1)[0]
                # ll = ll.mean()
                outs = Dict()

                outs.ctx_ll = ll[batch.mask_xc.int() == 1].mean()
                outs.tar_ll = ll[batch.mask_xt.int() == 1].mean()
            return outs

        # if self.deter_metric == 'RMSE':
        #     if num_samples is None:
        #         rmse = torch.sqrt(self.deter(p_yt.mean, Y_trgt))
        #     else:
        #         y = torch.stack([Y_trgt] * num_samples)
        #         rmse = torch.sqrt(self.deter(p_yt.mean, y))
        #
        #     if mask is not None:  # (b, H, W), reshape the middle two dims
        #         rmse = rmse * mask[None, :, :, None]
        #         rmse = torch.sum(rmse, dim=2)/torch.sum(mask, dim=-1)[0]
        #     else:
        #         num_ctx = X_cnxt.shape[-2]
        #         rmse = rmse[:, :, num_ctx:]
        #     rmse = rmse.mean()
        # return ll, rmse


class AdversarialLossGAN:
    def __init__(self):
        self.act = torch.nn.functional.sigmoid
    def __call__(self, pz, qz, x, y, xy_encoder, discriminator, n_sample=10, freeze_D=False):
        _, raw_encoded = xy_encoder(x, y, return_out=True)
        encoded = discriminator.encoded_linear(raw_encoded.detach())
        encoded = encoded[None, ...].expand([n_sample, -1, -1])
        z_post = qz.rsample([n_sample])
        post_ratio = discriminator(z_post, encoded)
        if freeze_D: # train the VAE, just min T(q(z), X, Y)
            post_score = torch.mean(post_ratio.squeeze(-1), dim=0)
            loss = torch.mean(post_score)
        else: # train D,  max log sigma[T(q(z), X, Y)] + log (1 - sigma((p(z), X, Y)))
            # post_score = torch.mean(1 + torch.log(post_ratio).squeeze(-1), dim=0)
            post_score = torch.log(self.act(post_ratio).squeeze(-1))
            post_score = torch.mean(post_score, dim=0)

            z_prior = pz.rsample([n_sample]).detach()
            prior_ratio = discriminator(z_prior, encoded)
            prior_score = torch.log(1- self.act(prior_ratio).squeeze(-1))
            prior_score = torch.mean(prior_score, dim=0)

            loss = -(post_score + prior_score)
            loss = torch.mean(loss)

        # # validation grad
        # loss.backward()
        # temp1 = x.grad
        # temp2 = z_post.grad
        # temp3 = encoded.grad
        # temp4 = discriminator.encoder.weight.grad
        return loss



class AdversarialLossKL:

    def __call__(self, pz, qz, x, y, xy_encoder, discriminator, n_sample=10, freeze_D=False):
        _, raw_encoded = xy_encoder(x, y, return_out=True)  # Shared DeepSet encoder

        encoded = discriminator.encoded_linear(raw_encoded.detach()) # discriminator decoder
        encoded = encoded[None, ...].expand([n_sample, -1, -1])

        z_post = qz.rsample([n_sample])
        if freeze_D: # train the VAE, just min T(q(z), X, Y)
            post_ratio = discriminator(z_post, encoded.detach())  # update only the posterior
            post_score = torch.mean(post_ratio.squeeze(-1), dim=0)
            loss = torch.mean(post_score)
        else: # train D,  min -T(q(z), X, Y)] +  exp(T(p(z), X, Y) - 1))
            # post_score = torch.mean(1 + torch.log(post_ratio).squeeze(-1), dim=0)
            post_ratio = discriminator(z_post, encoded) # update encoder as well
            post_score = post_ratio.squeeze(-1)
            post_score = torch.mean(post_score, dim=0)

            z_prior = pz.rsample([n_sample]).detach()
            prior_ratio = discriminator(z_prior, encoded)
            prior_score = torch.exp(prior_ratio -1).squeeze(-1)
            prior_score = torch.mean(prior_score, dim=0)

            loss = -(post_score - prior_score)
            loss = torch.mean(loss)

        # # validation grad
        # loss.backward()
        # temp1 = x.grad
        # temp2 = z_post.grad
        # temp3 = encoded.grad
        # temp4 = discriminator.encoder.weight.grad
        return loss

def plot_alpha_renyi_divergence():
    import numpy as np
    import matplotlib.pyplot as plt

    div = RenyiDivergence()
    mean_p = torch.rand((32, 2))
    mean_q = torch.rand((32, 2))
    var_p = torch.sigmoid(torch.rand((32, 2)))
    var_q = torch.sigmoid(torch.rand((32, 2)))
    q_zc = torch.distributions.Normal(mean_p, var_p)
    q_zt = torch.distributions.Normal(mean_q, var_q)

    kl_z = kl_divergence(q_zt, q_zc)
    kl_d = torch.mean(sum_from_nth_dim(kl_z, 1), dim=0).item()

    renyi_div = []
    alphas = torch.arange(0.01, 3.5, 0.05)
    for alpha in alphas:
        if torch.abs(alpha - 1.0) < 1e-4:
            continue
        else:
            renyi_d_alpha = div(0, q_zt, q_zc, alpha=alpha)
            renyi_div.append(torch.mean(renyi_d_alpha, dim=0).item())

    plt.plot(alphas.numpy(), np.array(renyi_div), label='Renyi Alpha Divergence')
    plt.scatter(x=1, y=kl_d, label='KLD', c='r')
    plt.legend()
    plt.savefig("Renyi_divergence.png")
    plt.show()


if __name__ == '__main__':
    plot_alpha_renyi_divergence()