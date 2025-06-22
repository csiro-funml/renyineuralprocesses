import os.path as osp
import torch
import numpy as np
import wget
import os
import sys
from addict import Dict
import matplotlib.pyplot as plt

# Path to directory A
# parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
# Add path to A to sys.path
# sys.path.append(parent_path)
# from utils.paths import datasets_path

def load_hare_lynx(num_batches, batch_size, parent_path=None, num_ctx=None):
    if parent_path is None:
        parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
    # print(datasets_path)
    datasets_path =''
    filename = osp.join(parent_path, datasets_path, 'lotka_volterra', 'LynxHare.txt')
    if not osp.isfile(filename):
        wget.download('http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt',
                out=osp.join(datasets_path, 'lotka_volterra'))

    tb = np.loadtxt(filename)
    times = torch.Tensor(tb[:,0]).unsqueeze(-1)
    pops = torch.stack([torch.Tensor(tb[:,2]), torch.Tensor(tb[:,1])], -1)

    # temporary plot
    # fig, ax = plt.subplots(figsize=(16, 4))
    # ax.scatter(times, pops[:, 0])
    # ax.scatter(times, pops[:, 1])
    # plt.show()

    #tb = pd.read_csv(osp.join(datasets_path, 'lotka_volterra', 'hare-lynx.csv'))
    #times = torch.Tensor(np.array(tb['time'])).unsqueeze(-1)
    #pops = torch.stack([torch.Tensor(np.array(tb['lynx'])),
    #    torch.Tensor(np.array(tb['hare']))], -1)

    batches = []
    N = pops.shape[-2]
    for _ in range(num_batches):
        batch = Dict()
        if num_ctx is None:
            num_ctx = torch.randint(low=15, high=N-15, size=[1]).item()
        num_tar = N - num_ctx

        idxs = torch.rand(batch_size, N).argsort(-1)

        batch.x = torch.gather(
                torch.stack([times]*batch_size),
                -2, idxs.unsqueeze(-1))
        batch.y = torch.gather(torch.stack([pops]*batch_size),
                -2, torch.stack([idxs]*2, -1))
        batch.xc = batch.x[:,:num_ctx]
        batch.xt = batch.x[:,num_ctx:]
        batch.yc = batch.y[:,:num_ctx]
        batch.yt = batch.y[:,num_ctx:]

        batches.append(batch)

    return batches



if __name__ == '__main__':
    # load
    batches = load_hare_lynx(1000, 16)
    #plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.scatter(batches[0].x[i, :, 0], batches[0].y[i, :, 0])
        ax.scatter(batches[0].x[i, :, 0], batches[0].y[i, :, 1])
    plt.show()