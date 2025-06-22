import torch
import torch.nn as nn
from torch.distributions import kl_divergence
# from attrdict import AttrDict
from addict import Dict
from utils.misc import stack, logmeanexp
from .modules import PoolingEncoder, Decoder
import math
import os


__all__ = ["NP"]

def to_numpy(x):
    return x.detach().cpu().numpy()

class NP(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            dim_lat=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3,
            use_mle=False,
            alpha=1.0):

        super().__init__()

        self.denc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.lenc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid,
                depth=dec_depth)

        self.use_mle = use_mle
        self.alpha = alpha

    def predict(self, xc, yc, xt, z=None, num_samples=None):
        theta = stack(self.denc(xc, yc), num_samples)
        if z is None:
            pz = self.lenc(xc, yc)
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        encoded = torch.cat([theta, z], -1)
        encoded = stack(encoded, xt.shape[-2], -2)
        return self.dec(encoded, stack(xt, num_samples))

    def sample(self, xc, yc, xt, z=None, num_samples=None):
        pred_dist = self.predict(xc, yc, xt, z, num_samples)
        return pred_dist.loc

    def predict_conditional_prior(self, xc, yc):
        pz = self.lenc(xc, yc)
        mu, sigma = pz.loc, pz.scale
        return (mu, sigma), pz

    def compute_robust_divergence(self, batch, py, z, pz, qz, num_samples=1):
        # K * B * N
        log_py = py.log_prob(stack(batch.y, num_samples)).sum(-1)
        # likelihood = to_numpy(recon)
        # print("likelihood", likelihood[0])

        log_pz = pz.log_prob(z).sum(-1)
        # prior = to_numpy(log_pz)
        # print("prior", prior)
        if qz is not None: # L_vi instead of L_ml
            # K * B
            log_qz = qz.log_prob(z).sum(-1)
            # poster = to_numpy(log_qz)
            # print("poster", poster)
            log_w = log_py.sum(-1) + log_pz - log_qz  # K * B
            if self.alpha == 1.0: # KL divergence
                bound = (torch.logsumexp(log_w, dim=0) - math.log(log_w.shape[0]))/py.loc.shape[-2]   # devided by n_target
                loss = -bound.mean() # batch average
            else:  # Renyi alpha divergence
                log_w = (1- self.alpha) * log_w
                bound = (torch.logsumexp(log_w, dim=0) - math.log(log_w.shape[0]))/((1- self.alpha)*py.loc.shape[-2])  # devided by n_target
                loss = -bound.mean() # batch average
        else: #L_ML
            log_w = log_py.sum(-1)
            bound = (torch.logsumexp(log_w, dim=0) - math.log(log_w.shape[0]))/py.loc.shape[-2] # devided by n_target
            loss = -bound.mean()
        return loss


    def compute_CRPS(self, py, y):
        # Ensure sigma > 0 to avoid NaNs
        mu = py.mean
        sigma = py.stddev

        eps = 1e-8
        sigma = sigma.clamp_min(eps)

        # Standardize
        z = (y - mu) / sigma

        # Get PDF and CDF of the standard normal
        dist = torch.distributions.Normal(torch.tensor(0.0, device=y.device), torch.tensor(1.0, device=y.device))
        pdf_z = torch.exp(dist.log_prob(z))
        cdf_z = dist.cdf(z)

        # CRPS formula for Normal
        # CRPS = sigma * [ z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ]
        crps = sigma * (
                z * (2 * cdf_z - 1.0)
                + 2.0 * pdf_z
                - 1.0 / math.sqrt(math.pi)
        )
        return crps


    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = Dict()
        if self.training:
            pz = self.lenc(batch.xc, batch.yc)
            qz = self.lenc(batch.x, batch.y)
            if not self.use_mle: # L_VI
                z = qz.rsample() if num_samples is None else \
                        qz.rsample([num_samples])
            else: # L_ML
                z = pz.rsample() if num_samples is None else \
                        pz.rsample([num_samples])
                qz = None
            py = self.predict(batch.xc, batch.yc, batch.x,
                    z=z, num_samples=num_samples)

            if num_samples > 1:
                loss = self.compute_robust_divergence(batch, py, z, pz, qz, num_samples=num_samples)
                outs.loss = loss
                outs.pz = pz
                outs.qz = qz
            else:
                outs.recon = py.log_prob(batch.y).sum(-1).mean()
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            # compute likelihood
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            else:
                y = torch.stack([batch.y]*num_samples)
                if reduce_ll:
                    ll = logmeanexp(py.log_prob(y).sum(-1))
                else:
                    ll = py.log_prob(y).sum(-1)

            # compute relative error
            relative_error = (torch.abs(py.mean - batch.y)/torch.abs(batch.y)).mean(-1) # (n_z, b, n_t+c)
            num_ctx = batch.xc.shape[-2]

            # compute Continuous Ranked Probability Score (CRPS)
            crps = self.compute_CRPS(py, batch.y)
            outs.crps = crps[..., num_ctx:, :].mean()
            if reduce_ll:
                outs.ctx_ll = ll[...,:num_ctx].mean()
                outs.tar_ll = ll[...,num_ctx:].mean()
                outs.ctx_re = relative_error[..., :num_ctx].mean()
                outs.tar_re = relative_error[..., num_ctx:].mean()
            else:
                outs.ctx_ll = ll[...,:num_ctx]
                outs.tar_ll = ll[...,num_ctx:]
                outs.ctx_re = relative_error[..., :num_ctx]
                outs.tar_re = relative_error[..., num_ctx:]
        return outs


