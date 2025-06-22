import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal, StudentT
from addict import Dict
import math
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import os
__all__ = ["GPPriorSampler", 'GPSampler', 'RBFKernel', 'PeriodicKernel', 'Matern52Kernel']


class GPPriorSampler(object):
    """
    Bayesian Optimization에서 이용
    """
    def __init__(self, kernel, t_noise=None, change_param=False, seed=None):
        self.kernel = self.select_kernel(kernel, change_param=change_param)
        self.t_noise = t_noise
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    def select_kernel(self, name, change_param=False):
        print(name)
        if name == 'RBF':
            if change_param:
                return RBFKernel( sigma_eps=2e-2, max_length=0.6, max_scale=2)
            else:
                return RBFKernel()
        elif name =='Periodic':
            if change_param:
                return PeriodicKernel(p=0.8)
            else:
                return PeriodicKernel()
        elif name == 'Matern':
            if change_param:
                return Matern52Kernel(sigma_eps=2e-2, max_length=0.6, max_scale=0.3)
            else:
                return Matern52Kernel()
        else:
            print("not supported kernels")
            exit(-1)

    # bx: 1 * num_points * 1
    def sample(self, x, device):
        # 1 * num_points * num_points
        cov = self.kernel(x)
        mean = torch.zeros(1, x.shape[1], device=device)

        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        if self.t_noise is not None:
            y += self.t_noise * StudentT(2.1).rsample(y.shape).to(device)

        return y


class GPSampler(object):
    def __init__(self, kernel, t_noise=None, seed=None, noise_strategy=None):
        self.kernel = self.select_kernel(kernel)
        self.t_noise = t_noise
        self.noise_strategy = noise_strategy
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    def select_kernel(self, name):
        print(name)
        if name == 'RBF':
            return RBFKernel()
        elif name =='Periodic':
            return PeriodicKernel()
        elif name == 'Matern':
            return Matern52Kernel()
        else:
            print("not supported kernels")
            exit(-1)

    def sample(self,
            batch_size=16,
            num_ctx=None,
            num_tar=None,
            max_num_points=50,
            x_range=(-2, 2),
            device='cpu'):

        batch = Dict()
        num_ctx = num_ctx or torch.randint(low=3, high=max_num_points-3, size=[1]).item()  # Nc
        num_tar = num_tar or torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()  # Nt

        num_points = num_ctx + num_tar  # N = Nc + Nt
        batch.x = x_range[0] + (x_range[1] - x_range[0]) \
                * torch.rand([batch_size, num_points, 1], device=device)  # [B,N,Dx=1]
        batch.xc = batch.x[:,:num_ctx]  # [B,Nc,1]
        batch.xt = batch.x[:,num_ctx:]  # [B,Nt,1]

        # batch_size * num_points * num_points
        cov = self.kernel(batch.x)  # [B,N,N]
        mean = torch.zeros(batch_size, num_points, device=device)  # [B,N]
        batch.y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)  # [B,N,Dy=1]
        batch.yc = batch.y[:,:num_ctx]  # [B,Nc,1]
        batch.yt = batch.y[:,num_ctx:]  # [B,Nt,1]

        if self.t_noise is not None:
            # if self.t_noise == -1:
            #     t_noise = 0.15 * torch.rand(batch.y.shape).to(device)  # [B,N,1]
            # else:
            #     t_noise = self.t_noise
            # batch.y += t_noise * StudentT(2.1).rsample(batch.y.shape).to(device)
            if self.noise_strategy == 'y':
                e_y = torch.randn(batch.yc.shape) * 1
                e_y = e_y.to(batch.yc.device)
                batch.yc = (1 - self.t_noise) * batch.yc + self.t_noise * e_y
            elif self.noise_strategy == 'x':
                e_x = torch.randn(batch.xc.shape) * 1
                e_x = e_x.to(batch.xc.device)
                batch.xc = (1 - self.t_noise) * batch.xc + self.t_noise * e_x
            else: # both x, y
                e_y = torch.randn(batch.yc.shape) * 1
                e_y = e_y.to(batch.yc.device)
                batch.yc = (1 - self.t_noise) * batch.yc + self.t_noise * e_y
                e_x = torch.randn(batch.xc.shape) * 1
                e_x = e_x.to(batch.xc.device)
                batch.xc = (1 - self.t_noise) * batch.xc + self.t_noise * e_x
        return batch
        # {"x": [B,N,1], "xc": [B,Nc,1], "xt": [B,Nt,1],
        #  "y": [B,N,1], "yc": [B,Nt,1], "yt": [B,Nt,1]}

    def sample_grid(self,
            num_ctx=None,
            num_tar=None,
            max_num_points=50,
            x_range=(-2, 2),
            device='cpu'):

        batch = Dict()
        num_ctx = num_ctx or torch.randint(low=3, high=max_num_points-3, size=[1]).item()  # Nc
        num_tar = num_tar or torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()  # Nt

        num_points = num_ctx + num_tar  # N = Nc + Nt
        x_grid = torch.linspace(x_range[0], x_range[1],  200, device=device).unsqueeze(0)
        batch.x_grid = x_grid.unsqueeze(-1)
        assert x_grid.shape[1] >= max_num_points
        # generate context and target idx
        shuffled_indices = torch.randperm(x_grid.shape[1])
        ctx_idx = shuffled_indices[:num_ctx]
        tgt_idx = shuffled_indices[num_ctx: num_ctx+num_tar]
        #

        # batch_size =1
        # num_points = 1000
        # batch.x_grid = x_range[0] + (x_range[1] - x_range[0]) \
        #         * torch.rand([batch_size, num_points, 1], device=device)  # [B,N,Dx=1]
        # batch.xc = batch.x_grid[:,:num_ctx]  # [B,Nc,1]
        # batch.xt = batch.x_grid[:, num_ctx:num_ctx+num_tar]  # [B,Nt,1]
        # batch.x = torch.cat([batch.xc, batch.xt], dim=1)
        # batch_size * num_points * num_points
        # cov = self.kernel(batch.x_grid)  # [B,N,N]
        # mean = torch.zeros(1, batch.x_grid.shape[1], device=device)
        cov = self.kernel(batch.x_grid)
        mean = torch.zeros(1, batch.x_grid.shape[1], device=device)# [B,N]

        batch.y_grid = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)  # [B,N,Dy=1]

        # batch.yc = batch.y_grid[:, ctx_idx]  # [B,Nc,1]
        # batch.yt = batch.y_grid[:, tgt_idx]  # [B,Nt,1]
        batch.xc = batch.x_grid[:, ctx_idx, :]  # [B,Nc,1]
        batch.xt = batch.x_grid[:, tgt_idx, :]  # [B,Nt,1]
        batch.x = torch.cat([batch.xc, batch.xt], dim=1)

        batch.yc =  batch.y_grid[:, ctx_idx]
        batch.yt = batch.y_grid[:, tgt_idx]
        batch.y = torch.cat([batch.yc, batch.yt], dim=1)
        if self.t_noise is not None:
            # if self.t_noise == -1:
            #     t_noise = 0.15 * torch.rand(batch.y.shape).to(device)  # [B,N,1]
            # else:
            #     t_noise = self.t_noise
            # batch.y += t_noise * StudentT(2.1).rsample(batch.y.shape).to(device)
            if self.noise_strategy == 'y':
                e_y = torch.randn(batch.yc.shape) * 1
                e_y = e_y.to(batch.yc.device)
                batch.yc = (1 - self.t_noise) * batch.yc + self.t_noise * e_y
            elif self.noise_strategy == 'x':
                e_x = torch.randn(batch.xc.shape) * 1
                e_x = e_x.to(batch.xc.device)
                batch.xc = (1 - self.t_noise) * batch.xc + self.t_noise * e_x
            else: # both x, y
                e_y = torch.randn(batch.yc.shape) * 1
                e_y = e_y.to(batch.yc.device)
                batch.yc = (1 - self.t_noise) * batch.yc + self.t_noise * e_y
                e_x = torch.randn(batch.xc.shape) * 1
                e_x = e_x.to(batch.xc.device)
                batch.xc = (1 - self.t_noise) * batch.xc + self.t_noise * e_x
        return batch
        # {"x": [B,N,1], "xc": [B,Nc,1], "xt": [B,Nt,1],
        #  "y": [B,N,1], "yc": [B,Nt,1], "yt": [B,Nt,1]}


class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim  [B,N,Dx=1]
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points * dim  [B,N,N,1]
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3))/length

        # batch_size * num_points * num_points  [B,N,N]
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov  # [B,N,N]

class Matern52Kernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = scale.pow(2)*(1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class PeriodicKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0, p=None):
        self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        if self.p is None:
            p = 0.1 + 0.4*torch.rand([x.shape[0], 1, 1], device=x.device)
        else:
            p = self.p
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(\
                - 2*(torch.sin(math.pi*dist.abs().sum(-1)/p)/length).pow(2)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov


def plot(name = 'RBF'):
    x = torch.linspace(-2, 2, 200)
    x = x[None, :, None]

    fig, axes = plt.subplots(2, 1, dpi=200, figsize=(6, 4))
    n_sample = 3
    gp_rbf = GPPriorSampler("RBF", change_param=True, seed=3)
    y_rbf = [gp_rbf.sample(x=x, device="cpu") for i in range(n_sample)]

    gp_matern = GPPriorSampler("Matern", change_param=True, seed=1)
    y_matern = gp_matern.sample(x=x, device="cpu")

    x = x.numpy()[0, :, 0]
    ax = axes[0]
    colors = cm.get_cmap('Blues', n_sample+3)  # 'Blues' colormap for steelblue shades

    for i in range(n_sample):
        y_i = y_rbf[i].numpy()[0, :, 0]
        label = '$D_{train}$' if i == 0 else ""
        ax.plot(x, y_i, label=label, linewidth=2, color=colors(n_sample-i+1))
    ax.legend()
    ax2 = axes[1]

    ax2.plot(x, y_matern.numpy()[0, :, 0], c='steelblue', linewidth=2, label="$D_{test}$")
    idx = torch.randperm(len(x))[:5]
    x_c = x[idx]
    y_c = y_matern.numpy()[0, idx, 0]
    # ax2.scatter(x_c, y_c, label='$C_{test}$')
    ax2.legend()
    # context set

    for i, ax in enumerate(axes):

        # Set major and minor tick locators for the x and y axis
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis

        ax.yaxis.set_major_locator(ticker.MultipleLocator(1 if i==0 else 0.2))  # Major ticks every 2 units on x-axis
        # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis

        font_size = 18

        ax.legend(loc='upper left', frameon=False, ncol=2,
                  bbox_to_anchor=(0, 1), borderaxespad=0.1,
                  fontsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)  # Set font size for x-tick labels
        ax.tick_params(axis='y', labelsize=font_size, )  # Set font size for y-tick labels

        # Increase the frame thickness (spines)
        for spine in ax.spines.values():
            spine.set_linewidth(3)  # Set the thickness of the frame (spines)

        # Increase the tick thickness
        ax.tick_params(axis='both', length=5)
    plt.tight_layout()
    parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
    plt.savefig(os.path.join(parent_path, "results/plots/illustration_priors.png"))
    plt.show()

    return


if __name__ == '__main__':
    plot("RBF")
    # plot("Matern")
