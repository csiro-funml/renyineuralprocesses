import os
import os.path as osp

import argparse
import yaml

import torch
import torch.nn as nn

import math
import pandas as pd
import time
import matplotlib.pyplot as plt
from addict import Dict
from tqdm import tqdm
from copy import deepcopy
from utils.load_module import load_module
from utils.paths import results_path, datasets_path, evalsets_path, config_path
from utils.log import get_logger, RunningAverage
from data.hare_lynx import load_hare_lynx  # TODO: go set your parent_path




def standardize(batch):
    with torch.no_grad():
        mu, sigma = batch.xc.mean(-2, keepdim=True), batch.xc.std(-2, keepdim=True)
        sigma[sigma==0] = 1.0
        batch.x = (batch.x - mu) / (sigma + 1e-5)
        batch.xc = (batch.xc - mu) / (sigma + 1e-5)
        batch.xt = (batch.xt - mu) / (sigma + 1e-5)

        mu, sigma = batch.yc.mean(-2, keepdim=True), batch.yc.std(-2, keepdim=True)
        batch.y = (batch.y - mu) / (sigma + 1e-5)
        batch.yc = (batch.yc - mu) / (sigma + 1e-5)
        batch.yt = (batch.yt - mu) / (sigma + 1e-5)
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
            choices=['train', 'eval',
                     'misspecified_eval', 'plot', 'gifplot'],
            default='plot')

    # Data
    parser.add_argument('--data_name', type=str, default='lotka_volterra', help='data set: lotka_volterra')
    # Model
    parser.add_argument('--model_name', type=str, default='NP', help='model selection: NP, ANP, TNPD')

    parser.add_argument('--divergence', type=str, default='Renyi_0.0',
                        help="Renyi_alpha and replace alpha with actual values")
    parser.add_argument('--scorerule', type=str, default='log', help="log")
    parser.add_argument('--train_seed', type=int, default=0, help='random seed')

    parser.add_argument('--expid', type=str, default='trial')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--max_num_points', type=int, default=50)

    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=32)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--hare_lynx', action='store_true')
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(config_path)
    with open(os.path.join(config_path, f'lotka_volterra/{args.model_name}.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    if args.model_name in ["NP", "ANP", "TNPD"]:
        alpha = float(args.divergence.split('_')[1]) # if alpha =0 use
        model = load_module(args.model_name, config, alpha=alpha)

    model.cuda() if torch.cuda.is_available() else None

    args.root = osp.join(results_path, args.data_name, args.model_name,
                         args.model_name + '_' + args.divergence + '_' + args.scorerule,
                         'run_' + str(args.train_seed))

    if args.mode == 'train':
        print("--- Training %s/%s_%s_%s/run_%d ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        train(args, model)
    elif args.mode == 'eval':
        print("--- Testing %s/%s_%s_%s/run_%d ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        eval(args, model)
    elif args.mode == "misspecified_eval":
        print("--- Testing %s/%s_%s_%s/run_%d with prior misspecification ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        misspecified_eval(args, model, load_model=True)
    elif args.mode == "plot":
        print("--- Plotting %s/%s_%s_%s/run_%d with prior misspecification ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        plot(args, model, load_model=True)
    elif args.mode == "gifplot":
        print("--- Plotting %s/%s_%s_%s/run_%d with prior misspecification ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        gifplot(args, model, load_model=True)


def train(args, model):
    if not osp.isdir(args.root):
        os.makedirs(args.root)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'train.tar'))
    eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))
    # eval_data = None
    num_steps = len(train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = osp.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))
    best_metric = -torch.inf
    for step in tqdm(range(start_step, num_steps+1)):
        model.train()
        optimizer.zero_grad()
        batch = standardize(train_data[step-1])
        for key, val in batch.items():
            batch[key] = val.to(device)

        outs = model(batch, num_samples=args.train_num_samples)
        loss = outs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        ravg.update('loss', loss.item())

        if step % args.print_freq == 0:
            line = f'{args.model_name}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line, _, _ = eval(args, model, eval_data=eval_data)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == num_steps:
            line, metric, df = eval(args, model,eval_data=eval_data, load_model=False)
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
                df.to_csv(os.path.join(args.root, "validation_set_metrics.csv"), index=False)


    args.mode = 'eval'
    eval(args, model, eval_data=eval_data)


def eval(args, model, eval_data=None, load_model=True):
    if eval_data is None:
        eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'eval':
        if load_model:
            ckpt = torch.load(os.path.join(args.root, 'model_params.pt'), map_location=device)
            model.load_state_dict(ckpt)
        if args.eval_logfile is None:
            if args.hare_lynx:
                eval_logfile = 'hare_lynx.log'
            else:
                eval_logfile = 'eval.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    # criterion = RenyiDivergence(alpha=args.divergence)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for batch in tqdm(eval_data):
            batch = standardize(batch)
            for key, val in batch.items():
                batch[key] = val.to(device)
            outs = model(batch, args.eval_num_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model_name}:{args.expid} '
    line += ravg.info()

    df = {}
    for key in ravg.sum.keys():
        val = ravg.sum[key] / ravg.cnt[key]
        df[key] = val
    df = pd.Series(df).to_frame().transpose()

    if logger is not None:
        logger.info(line)

    return line, df['tar_ll'].values[0], df


def plot(args, model, eval_data=None, load_model=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.font_manager as fm

    def tnp(x):
        return x.squeeze().cpu().data.numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'plot':
        if load_model:
            ckpt = torch.load(os.path.join(args.root, 'model_params.pt'), map_location=device)
            model.load_state_dict(ckpt)
        if args.eval_logfile is None:
            if args.hare_lynx:
                eval_logfile = 'hare_lynx.log'
            else:
                eval_logfile = 'eval.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    if eval_data is None:
        eval_data = load_hare_lynx(1000, 16, parent_path=datasets_path)  # use hare_lynx dataset

    model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_data):
            break

        batch = standardize(batch)
        for key, val in batch.items():
            batch[key] = val.to(device)

        batch_id = 10
        # batch_id = 1
        sort_value, sort_idx = torch.sort(batch.x[batch_id, :, 0])
        x_all = batch.x[[batch_id], sort_idx, :].unsqueeze(0)
        y_all = batch.y[[batch_id], sort_idx, :].unsqueeze(0)
        xc, yc = batch.xc[[batch_id]], batch.yc[[batch_id]]

        pred = model.predict(xc, yc, x_all, num_samples=args.eval_num_samples)

    mu, sigma = pred.mean.squeeze(), pred.scale.squeeze()

    # plot here
    fig, axes = plt.subplots(dpi=200, figsize=(6, 4))

    # plot here
    # fig, axes = plt.subplots(dpi=200, figsize=(10, 4))
    axes = [axes]
    # plot hare
    name_dict = {1: 'Hare', 0: 'Lynx'}
    # color_dict = {0: 'steelblue', 1: 'orange'}
    color_dict = {0: 'navy', 1: 'darkred'}
    predict_color_dict = {0: 'turquoise', 1: 'pink'}

    for i, ax in enumerate(axes):
        for n_species in range(y_all.shape[-1]):
            # ax.plot(tnp(x_all), tnp(y_all)[:, n_species],
            #         label='%s ground truth'%name_dict[n_species], color=color_dict[n_species],
            #         linewidth=2,
            #         zorder=1,
            #         ) # ground truth
            ax.plot(tnp(x_all), tnp(torch.mean(mu[:, :, n_species], dim=0)), color=color_dict[n_species],
                    label='%s prediction' % name_dict[n_species], linewidth=2.5, zorder=10,
                    alpha=0.4
                    )

            # predictions oversamples
            for s in range(mu.shape[0]):
                # ax.plot(tnp(x_all), tnp(mu[s, :, n_species]), color=predict_color_dict[n_species],
                #         alpha=max(0.5 / args.plot_num_samples, 0.7)
                #         )
                ax.fill_between(tnp(x_all), tnp(mu[s, :, n_species]) - tnp(sigma[s, :, n_species]),
                                tnp(mu[s, :, n_species]) + tnp(sigma[s, :, n_species]),
                                color=predict_color_dict[n_species],
                                alpha=0.01,
                                zorder=1,
                                # alpha=max(0.5/args.plot_num_samples, 2e-5),
                                linewidth=0.0)
            if len(tnp(yc).shape) == 1:
                ax.scatter(tnp(xc), tnp(yc)[n_species],
                           color=color_dict[n_species],
                           label='Observation set $S$' if n_species == 0 else "",
                           zorder=mu.shape[0] + 1, s=30,
                           marker='o' if n_species == 0 else 'x',

                           )  # context
            else:
                ax.scatter(tnp(xc), tnp(yc)[:, n_species],
                           color=color_dict[n_species],
                           label='%s context'%name_dict[n_species],
                           zorder=mu.shape[0] + 1, s=30,
                           marker='o' if n_species == 0 else 'x'
                           )  # context

        # Set major and minor tick locators for the x and y axis
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator())  # Minor ticks every 1 unit on x-axis
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Major ticks every 2 units on x-axis
        # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis

        font_size = 12
        # ax.set_title("n context: %d" % xc.shape[1], size=font_size)
        ax.legend(loc='upper left', frameon=False, ncol=2,
                  bbox_to_anchor=(0, 1), borderaxespad=0.1,
                  fontsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)  # Set font size for x-tick labels
        ax.tick_params(axis='y', labelsize=font_size, )  # Set font size for y-tick labels

        # Increase the frame thickness (spines)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # Set the thickness of the frame (spines)
        ax.set_ylim(-2, 4)
        # Increase the tick thickness
        ax.tick_params(axis='both', width=2.5)  # Set tick thickness
        ax.tick_params(axis='both', length=5)
        # ax.grid(True, linestyle="--", alpha=0.7)

    # axes = [axes]
    # # plot hare
    # name_dict = {1: 'Hare', 0: 'Lynx'}
    # # for i, ax in enumerate(axes):
    # for i in [0,1]:
    #     ax = axes[0]
    #     ax.plot(tnp(x_all), tnp(y_all)[:, i],
    #             label='%s ground truth'%name_dict[i], color='steelblue',
    #             linewidth=2,
    #             zorder=1,
    #             ) # ground truth
    #     ax.plot(tnp(x_all), tnp(torch.mean(mu[:, :, i], dim=0)), color='grey',
    #             label='predictive mean' if i == 0 else "",
    #             )
    #     ax.scatter(tnp(xc), tnp(yc)[:, i],
    #                color='k',
    #                label='context' , #if i ==1 else "",
    #                zorder=mu.shape[0] + 1,s=8
    #                ) # context
    #
    #
    #     # predictions oversamples
    #     for s in range(mu.shape[0]):
    #         ax.plot(tnp(x_all), tnp(mu[s, :,i]), color='grey',
    #                 alpha=max(0.5 / args.plot_num_samples, 0.1))
    #         ax.fill_between(tnp(x_all), tnp(mu[s, :, i]) - tnp(sigma[s, :, i]),
    #                         tnp(mu[s, :, i]) + tnp(sigma[s, :, i]),
    #                         color='lightgrey',
    #                         alpha=0.002,
    #                         # alpha=max(0.5/args.plot_num_samples, 2e-5),
    #                         linewidth=0.0)
    #     ax.legend()
    #
    #     # Set major and minor tick locators for the x and y axis
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
    #     # ax.xaxis.set_minor_locator(ticker.MultipleLocator())  # Minor ticks every 1 unit on x-axis
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Major ticks every 2 units on x-axis
    #     # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis
    #
    #     font_size = 17
    #
    #     ax.legend(loc='upper right', frameon=True, ncol = 2,
    #               bbox_to_anchor=(1,1), borderaxespad=0.1,
    #               fontsize=font_size)
    #     ax.tick_params(axis='x', labelsize=font_size)  # Set font size for x-tick labels
    #     ax.tick_params(axis='y', labelsize=font_size, )  # Set font size for y-tick labels
    #
    #     # Increase the frame thickness (spines)
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.5)  # Set the thickness of the frame (spines)
    #
    #     # Increase the tick thickness
    #     ax.tick_params(axis='both', width=4)  # Set tick thickness
    #     ax.tick_params(axis='both', length=5)
    #     # ax.grid(True, linestyle="--", alpha=0.7)
    #     # break
    plt.tight_layout()
    # plt.show()
    print("success")


    plt.savefig("results/plots/%s_%s_%s_%s_misspecification.png" % (args.model_name, args.data_name, args.divergence, args.train_seed))
    plt.show()


def gifplot(args, model, eval_data=None, load_model=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.font_manager as fm

    def tnp(x):
        return x.squeeze().cpu().data.numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'gifplot':
        if load_model:
            ckpt = torch.load(os.path.join(args.root, 'model_params.pt'), map_location=device)
            model.load_state_dict(ckpt)
        if args.eval_logfile is None:
            if args.hare_lynx:
                eval_logfile = 'hare_lynx.log'
            else:
                eval_logfile = 'eval.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    if eval_data is None:
        eval_data = load_hare_lynx(1000, 16, parent_path=datasets_path)  # use hare_lynx dataset

    model.eval()


    for n_context in range(1, 19):

        with torch.no_grad():
            for raw_batch in tqdm(eval_data):
                break

            batch = standardize(raw_batch)

            for key, val in batch.items():
                batch[key] = val.to(device)

            batch_id = 10
            sort_value, sort_idx = torch.sort(batch.x[batch_id, :, 0], descending=False)
            x_all = batch.x[[batch_id], sort_idx, :].unsqueeze(0)
            y_all = batch.y[[batch_id], sort_idx, :].unsqueeze(0)

            xc = x_all[:, torch.arange(0, n_context*5, 5), :]
            yc = y_all[:, torch.arange(0, n_context*5, 5), :]

            pred = model.predict(xc, yc, x_all, num_samples=10)

        mu, sigma = pred.mean.squeeze(), pred.scale.squeeze()

        # plot here
        fig, axes = plt.subplots(dpi=200, figsize=(10, 4))
        axes = [axes]
        # plot hare
        name_dict = {1: 'Hare', 0: 'Lynx'}
        color_dict = {0: 'steelblue', 1: 'orange'}
        predict_color_dict = {0: 'steelblue', 1: 'orange'}
        for i, ax in enumerate(axes):
            for n_species in range(y_all.shape[-1]):
                # ax.plot(tnp(x_all), tnp(y_all)[:, n_species],
                #         label='%s ground truth'%name_dict[n_species], color=color_dict[n_species],
                #         linewidth=2,
                #         zorder=1,
                #         ) # ground truth
                ax.plot(tnp(x_all), tnp(torch.mean(mu[:, :, n_species], dim=0)), color=color_dict[n_species],
                        label='%s prediction'%name_dict[n_species], linewidth = 2.5, zorder = 10
                        )

                # predictions oversamples
                for s in range(mu.shape[0]):
                    ax.plot(tnp(x_all), tnp(mu[s, :,n_species]), color=predict_color_dict[n_species],
                            alpha=max(0.5 / args.plot_num_samples, 0.7)
                            )
                    ax.fill_between(tnp(x_all), tnp(mu[s, :, n_species]) - tnp(sigma[s, :, n_species]),
                                    tnp(mu[s, :, n_species]) + tnp(sigma[s, :, n_species]),
                                    color=predict_color_dict[n_species],
                                    alpha=0.02,
                                    # alpha=max(0.5/args.plot_num_samples, 2e-5),
                                    linewidth=0.0)
                if len(tnp(yc).shape) == 1:
                    ax.scatter(tnp(xc), tnp(yc)[n_species],
                               color='k',
                               label='Observation set $S$' if n_species==0 else "",
                               zorder=mu.shape[0] + 1, s=8,
                               marker = 'o' if n_species == 0 else 'x'
                               )  # context
                else:
                    ax.scatter(tnp(xc), tnp(yc)[:, n_species],
                               color='k',
                               label='Observation set $S$'if n_species==0 else "",
                               zorder=mu.shape[0] + 1, s=8,
                               marker='o' if n_species == 0 else 'x'
                               )  # context

            # Set major and minor tick locators for the x and y axis
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 2 units on x-axis
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator())  # Minor ticks every 1 unit on x-axis
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Major ticks every 2 units on x-axis
            # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # Minor ticks every 1 unit on x-axis

            font_size = 12
            ax.set_title("n context: %d"%xc.shape[1], size=font_size)
            ax.legend(loc='upper left', frameon=False, ncol=2,
                      bbox_to_anchor=(0,1), borderaxespad=0.1,
                      fontsize=font_size)
            ax.tick_params(axis='x', labelsize=font_size)  # Set font size for x-tick labels
            ax.tick_params(axis='y', labelsize=font_size, )  # Set font size for y-tick labels

            # Increase the frame thickness (spines)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)  # Set the thickness of the frame (spines)
            ax.set_ylim(-2, 4)
            # Increase the tick thickness
            ax.tick_params(axis='both', width=2.5)  # Set tick thickness
            ax.tick_params(axis='both', length=5)
            ax.grid(True, linestyle="--", alpha=0.7)
            break
        plt.tight_layout()
        # plt.show()
        print("success")

        plt.savefig("results/plots/gif/lotka/n_context%d_%s_%s_%s_%s_misspecification.png" % (n_context, args.model_name, args.data_name, args.divergence, args.train_seed))
        plt.show()


def misspecified_eval(args, model, eval_data=None, load_model=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'misspecified_eval':
        if load_model:
            ckpt = torch.load(os.path.join(args.root, 'model_params.pt'), map_location=device)
            model.load_state_dict(ckpt)
        if args.eval_logfile is None:
            if args.hare_lynx:
                eval_logfile = 'hare_lynx.log'
            else:
                eval_logfile = 'eval.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    if eval_data is None:
        eval_data = load_hare_lynx(1000, 16, parent_path=datasets_path) # use hare_lynx dataset

    model.eval()
    ravg = RunningAverage()
    # criterion = RenyiDivergence(alpha=args.divergence)

    with torch.no_grad():
        for batch in tqdm(eval_data):
            batch = standardize(batch)
            for key, val in batch.items():
                batch[key] = val.to(device)
            outs = model(batch, args.eval_num_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model_name}:{args.expid} '
    line += ravg.info()

    df = {}
    for key in ravg.sum.keys():
        val = ravg.sum[key] / ravg.cnt[key]
        df[key] = val
    df = pd.Series(df).to_frame().transpose()

    if logger is not None:
        logger.info(line)

    # replace the data_name to hare_lynx
    save_name = osp.join(results_path, 'hare_lynx', args.model_name,
             args.model_name + '_' + args.divergence + '_' + args.scorerule,
             'run_' + str(args.train_seed))
    os.makedirs(save_name, exist_ok=True)
    df.to_csv(os.path.join(save_name, "validation_set_metrics.csv"), index=False)
    return line, df['tar_ll'].values[0], df

if __name__ == '__main__':
    main()