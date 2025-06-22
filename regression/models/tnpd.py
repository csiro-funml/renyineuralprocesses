import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from addict import Dict
import numpy as np
from models.tnp import TNP


class TNPD(TNP):
    def __init__(
            self,
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std=False,
            use_mle=False,
            alpha=0.0,
    ):
        super(TNPD, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std
        )

        self.n_z = 0 if use_mle else 32 # q(z) reparameterization
        self.predictor = nn.Sequential(
            nn.Linear(d_model+self.n_z, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y * 2)
        )
        self.use_mle = use_mle
        self.alpha = alpha

    def forward(self, batch, num_samples=1, reduce_ll=True):
        if self.training:
            if not self.use_mle:
                z_target = self.encode(batch, autoreg=False, include_xc=False)
                q_z = torch.rand((num_samples, *z_target.shape[:-1], self.n_z)).to(batch.xc.device) # reparameterization trick
                z_target_aug = z_target.unsqueeze(0).repeat(num_samples, 1, 1, 1)
                z_target_concat = torch.cat([z_target_aug, q_z], dim=-1)
                out = self.predictor(z_target_concat)
                # out = out.unsqueeze(0)  # n_samples =1
                mean, std = torch.chunk(out, 2, dim=-1)
                if self.bound_std:
                    std = 0.05 + 0.95 * F.softplus(std)
                else:
                    std = torch.exp(std)

                pred_tar = Normal(mean, std)

                ll = pred_tar.log_prob(batch.yt).sum(-1)
                if self.alpha!=1.0:
                    ll = (1-self.alpha) * ll.sum(-1)

                    ll = torch.logsumexp(ll, dim=0) - np.log(num_samples)

                    ll = ll / ((1- self.alpha) * batch.xt.shape[-2])
                else:
                    ll = ll.sum(-1)
                    ll = torch.logsumexp(ll, dim=0) - np.log(num_samples)
                    ll = ll / (batch.xt.shape[-2])
                #
                outs = Dict()
                outs.loss = -ll.mean()
            else:
                z_target = self.encode(batch, autoreg=False, include_xc=False)
                out = self.predictor(z_target)
                out = out.unsqueeze(0)  # n_samples =1
                mean, std = torch.chunk(out, 2, dim=-1)
                if self.bound_std:
                    std = 0.05 + 0.95 * F.softplus(std)
                else:
                    std = torch.exp(std)

                pred_tar = Normal(mean, std)
                #
                outs = Dict()
                if reduce_ll:
                    outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
                else:
                    outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
                outs.loss = - (outs.tar_ll)
            return outs

        else:
            outs = Dict()
            if self.use_mle:
                num_samples = 1
                pred_tar = self.predict(batch.xc, batch.yc, batch.x, num_samples)  # include both the target and context
                ll = pred_tar.log_prob(batch.y).sum(-1)
            else:

                pred_tar = self.predict(batch.xc, batch.yc, batch.x, num_samples)  # include both the target and context
                ll = pred_tar.log_prob(batch.y).sum(-1)
                ll = torch.logsumexp(ll, dim=0) - np.log(num_samples)

            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_ll = ll[..., :num_ctx].mean()
                outs.tar_ll = ll[..., num_ctx:].mean()
            else:
                outs.ctx_ll = ll[..., :num_ctx]
                outs.tar_ll = ll[..., num_ctx:]
            return outs

    def predict(self, xc, yc, xt, num_samples):
        batch = Dict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device=batch.xc.device)
        print("num samples", num_samples)
        z_target = self.encode(batch, autoreg=False)
        q_z = torch.rand((num_samples, *z_target.shape[:-1], self.n_z)).to(batch.xc.device)  # reparameterization trick
        z_target_aug = z_target.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        z_target_concat = torch.cat([z_target_aug, q_z], dim=-1)

        out = self.predictor(z_target_concat)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)
