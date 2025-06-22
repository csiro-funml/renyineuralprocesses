import os
import os.path as osp
import argparse
import yaml
import torch
import math
import numpy as np
import pandas as pd
import time
import csv
from addict import Dict
from tqdm import tqdm
from data.gp import *
from utils.load_module import load_module
from utils.log import get_logger, RunningAverage
from utils.paths import results_path, evalsets_path, config_path  # TODO: to replace them with your path



def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_name', type=str, default='RBF', help='data set: RBF, Periodic, Matern')
    # Model
    parser.add_argument('--model_name', type=str, default='NP', help='model selection: NP, ANP, TNPD')

    parser.add_argument('--train_seed', type=int, default=0, help='random seed')
    parser.add_argument('--divergence', type=str, default='Renyi_0.7', help="Renyi_alpha and replace alpha with actual values") # Renyi_1.0
    parser.add_argument('--scorerule', type=str, default='log', help="log")

    # Experiment
    parser.add_argument('--mode', default='plot', choices=['train', 'eval', 'plot'])
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)

    # Data
    parser.add_argument('--max_num_points', type=int, default=50)

    # Train
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000) # how often we eval and save model

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000) # number of batches in the evaluation set
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    args.train_seed = int(args.train_seed)

    args.root = osp.join(results_path, args.data_name, args.model_name, args.model_name + '_' + args.divergence + '_' + args.scorerule,
                         'run_' + str(args.train_seed))
    print(args)
    print(config_path)

    with open(os.path.join(config_path, f'gp/{args.model_name}.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    if args.model_name in ["NP", "ANP", "ConvNP", "BANP", "TNPD"]:
        div = args.divergence.split('_')
        alpha = float(div[1]) if len(div) >1  else div[0] # if alpha =0 use
        model = load_module(args.model_name, config, alpha=alpha, n_z_samples_test=args.eval_num_samples)
    else: # NP_TwoPQ use alpha =1.0
        model = load_module(args.model_name, config, alpha=1.0, n_z_samples_test=args.eval_num_samples)


    model.cuda() if torch.cuda.is_available() else None

    if args.mode == 'train':
        print("--- Training %s/%s_%s_%s/run_%d ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        train(args, model)
    elif args.mode == 'eval':
        print("--- Testing %s/%s_%s_%s/run_%d ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        eval(args, model, load_model=True)
    else:
        print("--- Plotting %s/%s_%s_%s/run_%d ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        plot(args, model, load_model=True)
    print("finished")

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        print(args.root + "exists")
        # if args.resume is None:
        #     raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    sampler = GPSampler(kernel=args.data_name, t_noise=args.t_noise, seed=args.eval_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(args.root,
                                   f'train.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.model_name}-{args.expid}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    best_metric = -torch.inf
    # Record the start time
    start_time = time.time()

    for step in tqdm(range(start_step, args.num_steps + 1)):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        outs = model(batch, num_samples=args.train_num_samples)
        loss = outs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        ravg.update('loss', loss.item())

        if step % args.print_freq == 0:
            line = f'{args.model_name}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line, metric = eval(args, model)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            line, metric = eval(args, model, load_model=False)
            logger.info(line + '\n')
            if metric > best_metric:
                logger.info("save model at epoch %d" % step)
                best_metric = metric
                ckpt = Dict()
                ckpt.model = model.state_dict()
                ckpt.optimizer = optimizer.state_dict()
                ckpt.scheduler = scheduler.state_dict()
                ckpt.logfilename = logfilename
                ckpt.step = step + 1
                torch.save(model.state_dict(), os.path.join(args.root, 'model_params.pt'))

    # Record the overall elapsed time
    overall_time = time.time() - start_time

    # Save to CSV
    # Prepare a CSV file to log the overall time
    csv_file = os.path.join(args.root,"overall_time.csv")
    fields = ["Overall Time (seconds)"]
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)  # Write the header
        writer.writerow([overall_time])  # Write the overall time

    print(f"Overall time saved to {csv_file}: {overall_time:.2f} seconds")
    # args.mode = 'eval'
    # print("--- Testing %s/%s_%s_%s/run_%d ---" % (
    #     args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
    # eval(args, model)


def get_eval_path(args):
    path = osp.join(evalsets_path, 'gp')
    filename = f'{args.data_name}-seed0'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    return path, filename


def gen_evalset(args):
    if args.data_name == 'RBF':
        kernel = RBFKernel()
    elif args.data_name == 'Matern':
        kernel = Matern52Kernel()
    elif args.data_name == 'Periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.data_name}')
    print(f"Generating Evaluation Sets with {args.data_name} kernel")

    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_points=args.max_num_points,
            device='cuda' if torch.cuda.is_available() else 'cpu'))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))


def eval(args, model, load_model=True):
    # eval a trained model on log-likelihood
    if args.mode == 'eval':
        if load_model:
            ckpt = torch.load(os.path.join(args.root, 'model_params.pt'),
                              map_location='cuda' if torch.cuda.is_available() else "cpu")
            model.load_state_dict(ckpt)
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.data_name}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename), map_location='cuda' if torch.cuda.is_available() else "cpu")

    if args.mode == "eval":
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed(args.train_seed)

    ravg = RunningAverage()
    model.eval()
    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in eval_batches:
            for key, val in batch.items():
                batch[key] = val.cuda() if torch.cuda.is_available() else val
                outs = model(batch, args.eval_num_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    # torch.cuda.manual_seed(time.time())

    line = f'{args.model_name}:{args.expid} {args.data_name} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    df = {}
    for key in ravg.sum.keys():
        val = ravg.sum[key] / ravg.cnt[key]
        df[key] = val
    df = pd.Series(df).to_frame().transpose()
    df.to_csv(os.path.join(args.root, "validation_set_metrics.csv"), index=False)
    # logger.info(line + '\n')
    print(line)
    return line, df['tar_ll'].values[0]

def plot(args, model, load_model =True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.font_manager as fm

    seed = args.eval_seed
    num_smp = 50
    if args.data_name == 'Periodic':
        args.plot_num_ctx = 60
        args.plot_num_tar = 10
    else:
        args.plot_num_ctx = 15
        args.plot_num_tar = 10
    args.plot_batch_size =1
    args.plot_num_samples = num_smp


    device = 'cuda' if torch.cuda.is_available() else "cpu"
    if load_model:
        ckpt = torch.load(os.path.join(args.root, 'model_params.pt'),
                          map_location='cuda' if torch.cuda.is_available() else "cpu")
        model.load_state_dict(ckpt)
        print("loaded weights")

    def tnp(x):
        return x.squeeze().cpu().data.numpy()


    # sampler = GPSampler(args.data_name, t_noise=args.t_noise, seed=args.eval_seed)

    xp = torch.linspace(-2, 2, 200)[None,:,None].to(device)

    sampler = GPPriorSampler(args.data_name, seed=2, change_param=False)
    yp = sampler.sample(xp, device=device)

    # randomly select context point
    indices = torch.randperm(xp.shape[1])
    batch = Dict()
    batch.xc = xp[:, indices[:args.plot_num_ctx], :]
    batch.yc = yp[:, indices[:args.plot_num_ctx], :]
    batch.xt = xp[:, indices[args.plot_num_ctx: args.plot_num_ctx+ args.plot_num_tar], :]
    batch.yt = yp[:, indices[args.plot_num_ctx: args.plot_num_ctx + args.plot_num_tar], :]
    batch.x = torch.cat([batch.xc, batch.xt], dim=1)
    batch.y = torch.cat([batch.yc, batch.yt], dim=1)
    batch.x_grid = xp
    batch.y_grid = yp
    # batch = sampler.sample_grid(
    #     num_ctx=args.plot_num_ctx,
    #     num_tar=args.plot_num_tar,
    #     device=device,
    # )
    # batch = sampler.sample(
    #     batch_size=1,
    #     num_ctx=args.plot_num_ctx,
    #     num_tar=args.plot_num_tar,
    #     device=device,
    # )


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Nc = batch.xc.size(1)
    Nt = batch.xt.size(1)

    model.eval()
    with torch.no_grad():
        outs = model(batch, num_smp, reduce_ll=False)
        tar_loss = outs.tar_ll  # [Ns,B,Nt] ([B,Nt] for CNP)

        xt = xp.repeat(args.plot_batch_size, 1, 1)
        pred = model.predict(batch.xc, batch.yc, xt, num_samples=num_smp)
        mu, sigma = pred.mean, pred.scale

    grid_x, grid_y = tnp(batch.x_grid), tnp(batch.y_grid)
    sort_idx = np.argsort(grid_x)
    grid_x = grid_x[sort_idx]
    grid_y = grid_y[sort_idx]
    fig, ax = plt.subplots(dpi=200, figsize=(6, 4))
    axes = [ax]


    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            ax.plot(grid_x, grid_y, color='steelblue',
                    label='ground truth',
                    linewidth=2,
                    zorder=1,
                    )
            ax.plot(tnp(xp), tnp(torch.mean(mu, dim=0)), color='grey',
                    label='predictive mean',
                    )
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='grey',
                    alpha=max(0.5/args.plot_num_samples, 0.1))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='lightgrey',
                        alpha=0.002,
                        # alpha=max(0.5/args.plot_num_samples, 2e-5),
                        linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'n context ={Nc}', zorder=mu.shape[0] + 1,
                       s=8)
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='darkred', label=f'n target ={Nt}',
                       s=20, marker='x',
                       zorder=mu.shape[0] + 1)
            ax.legend()
            # ax.set_title(f"tar_ll: {tar_loss[:, i, :].mean(): 0.4f}" + "   seed: %d"%args.train_seed +  "    " + args.divergence)
            # Set major and minor tick locators for the x and y axis
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis

            # ax.set_xlim(-2, 2)
            # ax.set_ylim(-2.5, 2.2)
            # if args.data_name != 'Matern':
            #     ax.set_ylim(-1.3, 1.3)
            # else:
            #     ax.set_ylim(-0.5, 0.5)
            #     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            font_size = 12

            ax.legend(loc='upper left', frameon=False,ncol=2,
                      bbox_to_anchor=(0, 1), borderaxespad=0.1,
                      fontsize=font_size)
            ax.tick_params(axis='x', labelsize=font_size)  # Set font size for x-tick labels
            ax.tick_params(axis='y', labelsize=font_size, )  # Set font size for y-tick labels


            # Increase the frame thickness (spines)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)  # Set the thickness of the frame (spines)

            # Increase the tick thickness
            ax.tick_params(axis='both', width=2.5)  # Set tick thickness
            ax.tick_params(axis='both', length=5)
            # ax.grid(True, linestyle="--", alpha=0.7)
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}')
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}')
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")

    # plt.suptitle(f"{args.expid}", y=0.995)
    plt.tight_layout()
    os.makedirs("results/plots/", exist_ok=True)
    plt.savefig(os.path.join(os.path.curdir,"results/plots/%s_%s_%s_%s.png"%(args.model_name, args.data_name, args.divergence, args.train_seed)))

    # save data to pt
    # torch.save(batch, os.path.join(os.path.curdir,"results/plots/%s_%s.pt"%(args.data_name, args.train_seed)))
    plt.show()



if __name__ == '__main__':
    main()