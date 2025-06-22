import torch
import numpy as np
import numpy.random as npr
import numba as nb
from tqdm import tqdm
from addict import Dict
#import pandas as pd
import wget
import os.path as osp
import os
import sys

# Path to directory A
parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
# Add path to A to sys.path
sys.path.append(parent_path)
from utils.paths import datasets_path



#https://jckantor.github.io/CBE30338/02.05-Hare-and-Lynx-Population-Dynamics.html#

@nb.njit(nb.i4(nb.f8[:]))
def catrnd(prob):
    cprob = prob.cumsum()
    u = npr.rand()
    for i in range(len(cprob)):
        if u < cprob[i]:
            return i
    return i

# @nb.njit(nb.types.Tuple((nb.f8[:,:,:], nb.f8[:,:,:], nb.i4)) \
#         (nb.i4, nb.i4, nb.i4, \
#         nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.b1))
def _simulate_task(batch_size, num_steps, max_num_points,
        X0, Y0, theta0, theta1, theta2, theta3, return_pop=False):

    time = np.zeros((batch_size, num_steps, 1))
    pop = np.zeros((batch_size, num_steps, 2))
    length = num_steps*np.ones((batch_size))

    for b in range(batch_size):
        pop[b,0,0] = max(int(X0 + npr.randn()), 1)
        pop[b,0,1] = max(int(Y0 + npr.randn()), 1)
        for i in range(1, num_steps):
            X, Y = pop[b,i-1,0], pop[b,i-1,1]
            rates = np.array([
                theta0*X*Y,
                theta1*X,
                theta2*Y,
                theta3*X*Y])
            total_rate = rates.sum()

            time[b,i,0] = time[b,i-1,0] + npr.exponential(scale=1./total_rate)

            pop[b,i,0] = pop[b,i-1,0]
            pop[b,i,1] = pop[b,i-1,1]
            a = catrnd(rates/total_rate)
            if a == 0:
                pop[b,i,0] += 1
            elif a == 1:
                pop[b,i,0] -= 1
            elif a == 2:
                pop[b,i,1] += 1
            else:
                pop[b,i,1] -= 1

            if pop[b,i,0] == 0 or pop[b,i,1] == 0:
                length[b] = i+1
                break

    if return_pop:
        return pop
    num_ctx = npr.randint(15, max_num_points-15)
    num_tar = npr.randint(15, max_num_points-num_ctx)
    num_points = num_ctx + num_tar
    min_length = length.min()
    while num_points > min_length:
        num_ctx = npr.randint(15, max_num_points-15)
        num_tar = npr.randint(15, max_num_points-num_ctx)
        num_points = num_ctx + num_tar

    x = np.zeros((batch_size, num_points, 1))
    y = np.zeros((batch_size, num_points, 2))
    for b in range(batch_size):
        idxs = np.arange(int(length[b]))
        npr.shuffle(idxs)
        for j in range(num_points):
            x[b,j,0] = time[b,idxs[j],0]
            y[b,j,0] = pop[b,idxs[j],0]
            y[b,j,1] = pop[b,idxs[j],1]

    return x, y, num_ctx

class LotkaVolterraSimulator(object):
    def __init__(self,
            X0=50,
            Y0=100,
            theta0=0.01,
            theta1=0.5,
            theta2=1.0,
            theta3=0.01):

        self.X0 = X0
        self.Y0 = Y0
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def simulate_tasks(self,
            num_batches,
            batch_size,
            num_steps=20000,
            max_num_points=100):

        batches = []
        for _ in tqdm(range(num_batches)):

            x, y, num_ctx = _simulate_task(
                    batch_size, num_steps, max_num_points,
                    self.X0, self.Y0, self.theta0, self.theta1, self.theta2, self.theta3)
            batch = Dict()
            batch.x = torch.Tensor(x)
            batch.y = torch.Tensor(y)
            batch.xc = batch.x[:,:num_ctx]
            batch.xt = batch.x[:,num_ctx:]
            batch.yc = batch.y[:,:num_ctx]
            batch.yt = batch.y[:,num_ctx:]

            batches.append(batch)

        return batches


    def sample_one_traject(self, batch_size=1, num_steps=6000, max_num_points=20000):
        pop = _simulate_task(batch_size, num_steps, max_num_points,
             self.X0, self.Y0, self.theta0, self.theta1, self.theta2, self.theta3,return_pop=True)[0]

        idxs = np.arange(pop.shape[0])
        npr.shuffle(idxs)

        n_context = 50
        pop_context = pop[idxs[:n_context]]

        # plot here
        fig, axes = plt.subplots(dpi=200, figsize=(10, 4))
        axes = [axes]
        # plot hare
        name_dict = {1: 'Predator', 0: 'Prey'}
        for i, ax in enumerate(axes):
            ax.plot(pop[:, 0],
                    label='%s' % name_dict[0],
                    linewidth=2,
                    zorder=1,
                    )  # ground truth
            ax.plot(pop[:, 1],
                    label='%s' % name_dict[1],
                    linewidth=2,
                    zorder=1,
                    )  # ground truth

            ax.scatter(idxs[:n_context], pop_context[:, 0],
                       color='k',
                       label='Observation set $S$',  # if i ==1 else "",
                       s=20
                       )  # context

            ax.scatter(idxs[:n_context], pop_context[:, 1],
                       color='k',
                       # label='$S$',  # if i ==1 else "",
                       s=20
                       )  # context
            ax.legend()

            # Set major and minor tick locators for the x and y axis
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator())  # Minor ticks every 1 unit on x-axis
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Major ticks every 2 units on x-axis
            # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis

            font_size = 12

            ax.legend(loc='upper left', frameon=True, ncol=2,
                      bbox_to_anchor=(0, 1), borderaxespad=0.1,
                      fontsize=font_size)
            # ax.tick_params(axis='x', labelsize=font_size)  # Set font size for x-tick labels
            # ax.tick_params(axis='y', labelsize=font_size, )  # Set font size for y-tick labels

            # Increase the frame thickness (spines)
            # for spine in ax.spines.values():
            #     spine.set_linewidth(1.5)  # Set the thickness of the frame (spines)
            ax.set_xlabel("$x$", fontdict={'size': font_size})
            ax.set_ylabel("$y$", fontdict={'size': font_size})
            # Increase the tick thickness
            ax.tick_params(axis='both', width=2.5)  # Set tick thickness
            ax.tick_params(axis='both', length=5)
            ax.grid(True, linestyle="--", alpha=0.7)
            break
        plt.tight_layout()
        # plt.show()
        print("success")
        # plt.savefig("../results/plots/gif/lotka/train.png")
        # plt.plot(pop[0, :, 0])
        # plt.plot(pop[0, :, 1])
        plt.show()
        return


if __name__ == '__main__':
    import argparse
    import os
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=1000) # eval with 1000 sampples , train with 10000
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--filename', type=str, default='train')
    parser.add_argument('--X0', type=float, default=50)
    parser.add_argument('--Y0', type=float, default=100)
    parser.add_argument('--theta0', type=float, default=0.01)
    parser.add_argument('--theta1', type=float, default=0.5)
    parser.add_argument('--theta2', type=float, default=1.0)
    parser.add_argument('--theta3', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--trajectory_all', type=int, default=0)

    args = parser.parse_args()

    sim = LotkaVolterraSimulator(X0=args.X0, Y0=args.Y0,
            theta0=args.theta0, theta1=args.theta1,
            theta2=args.theta2, theta3=args.theta3)

    trajectory_all = args.trajectory_all
    if trajectory_all:
        traj = sim.sample_one_traject()
    else: # generate the training validation batches
        root = os.path.join(parent_path, datasets_path, 'lotka_volterra')
        os.makedirs(root, exist_ok=True)

        if os.path.exists(os.path.join(root, f'{args.filename}.tar')):
            batches = torch.load(os.path.join(root, f'{args.filename}.tar'))
        else:
            batches = sim.simulate_tasks(args.num_batches, args.batch_size,
                                         num_steps=args.num_steps)

        fig, axes = plt.subplots(1, 4, figsize=(16,4))
        for i, ax in enumerate(axes.flatten()):
            ax.scatter(batches[0].x[i,:,0], batches[0].y[i,:,0])
            ax.scatter(batches[0].x[i,:,0], batches[0].y[i,:,1])
        plt.show()