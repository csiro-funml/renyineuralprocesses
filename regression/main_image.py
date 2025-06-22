import os
import os.path as osp
import argparse
import yaml
import torch
import tensorflow as tf # for plotting purpose, comment if unneeded
import numpy as np
import time
import csv
from addict import Dict
from tqdm import tqdm
import pandas as pd
from data.image import img_to_task, task_to_img, half_img_to_task
from data.mnist import MNIST
from data.celeba import CelebA32
from data.svhn import SVHN
from utils.load_module import load_module
from utils.paths import results_path, evalsets_path, config_path
from utils.log import get_logger, RunningAverage
import math



def convnp_batch_preprocessing(img, max_num_points=None):
    B, C, H, W = img.shape
    num_pixels = H * W
    raw_img = img.clone()
    img = img.view(B, C, -1)

    # Step 1: Create an M-by-N zero matrix
    mask = torch.zeros((H, W)).to(img.device)

    # Step 2: Flatten the matrix to a 1D tensor
    flattened_mask = mask.view(-1)

    # Step 3: Randomly select P indices from the 1D tensor
    max_num_points = max_num_points or num_pixels
    num_ctx = torch.randint(low=3, high=max_num_points - 3, size=[1]).item()
    indices = torch.randperm(flattened_mask.numel())[:num_ctx]

    # Step 4: Set the selected indices to 1
    flattened_mask[indices] = 1

    # Step 5: Reshape the tensor back to an M-by-N matrix
    mask_xc = flattened_mask.view(H, W)
    mask_xc = mask_xc.unsqueeze(0).repeat((B, 1, 1)) # B, H, W, add batch_size
    mask_xt = 1 - mask_xc

    batch = Dict()

    batch.xc = mask_xc.unsqueeze(-1).bool() #B, H, W, C # add channel
    batch.xt = mask_xt.unsqueeze(-1).bool() #B, H, W, C
    batch.yc = batch.xc * raw_img.permute(0, 2, 3, 1)
    batch.yt = batch.xt * raw_img.permute(0, 2, 3, 1)

    batch.x = torch.ones_like(batch.xt)
    batch.y = raw_img.permute(0, 2, 3, 1)

    batch.mask_xc = mask_xc
    batch.mask_xt = mask_xt
    return batch


def select_dataset(args, split='train'):
    if args.data_name == 'mnist':
        datasets = MNIST(split=split)
    elif args.data_name == 'celeba':
        datasets = CelebA32(split=split, truncate_dataset=args.model_name=='ConvNP')
    elif args.data_name == 'svhn':
        datasets = SVHN(split=split)
    else:
        print("dataset not defined")
        exit(-1)
    return datasets

def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_name', type=str, default='mnist', help='data set: mnist, svhn, celeba')
    parser.add_argument('--max_num_points', type=int, default=200)

    # Experiment
    parser.add_argument('--mode', choices=['train', 'eval', 'plot'],
                        default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)

    # Model
    parser.add_argument('--model_name', type=str, default='NP', help='model selection: NP, ANP, ConvNP, BANP, TNPD')
    parser.add_argument('--train_seed', type=int, default=3, help='random seed')
    parser.add_argument('--divergence', type=str, default='Renyi_1.0', help="Renyi_alpha")
    parser.add_argument('--scorerule', type=str, default='log', help="log")

    # Train
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=32)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=5)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()

    args.root = osp.join(results_path, args.data_name, args.model_name,
                         args.model_name + '_' + args.divergence + '_' + args.scorerule,
                         'run_' + str(args.train_seed))
    print("root", args.root)
    print(config_path)

    if "_TwoPQ" in args.model_name:
        pretrain_model_name = args.model_name.split('_')[0]
        with open(os.path.join(config_path, args.data_name,f'{pretrain_model_name}.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(os.path.join(config_path, args.data_name, f'{args.model_name}.yaml'), 'r') as f:
            config = yaml.safe_load(f)

    if args.model_name in ["NP", "ANP", "ConvNP", "BANP", "TNPD"]:
        alpha = float(args.divergence.split('_')[1])  # if alpha =0 use
        args.alpha = alpha
        model = load_module(args.model_name, config, alpha=alpha, n_z_samples_test=args.eval_num_samples)
    else:
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
    elif args.mode == 'plot':
        print("--- Plotting %s/%s_%s_%s/run_%d ---" % (
            args.data_name, args.model_name, args.divergence, args.scorerule, args.train_seed))
        plot(args, model)
    print("finished")

def train(args, model):
    os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    train_ds = select_dataset(args,split="train")
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.train_batch_size,
                                               shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))
    best_metric = -torch.inf
    # Record the start time
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in tqdm(range(start_epoch, args.num_epochs + 1)):
        model.train()
        for (x, _) in tqdm(train_loader, ascii=True):
            x = x.to(device)
            if args.model_name == 'ConvNP':
                batch = convnp_batch_preprocessing(x, args.max_num_points)
            else:
                batch = img_to_task(x,
                                max_num_points=args.max_num_points,
                                model_name=args.model_name)
            optimizer.zero_grad()

            outs = model(batch, num_samples=args.train_num_samples)
            loss = outs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            ravg.update('loss', loss.item())

        line = f'{args.model_name}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            line, metric = eval(args, model)
            logger.info(line + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            line, metric = eval(args, model, load_model=False)
            logger.info(line + '\n')
            if metric > best_metric:
                logger.info("save model at epoch %d" % epoch)
                best_metric = metric
                ckpt = Dict()
                ckpt.model = model.state_dict()
                ckpt.optimizer = optimizer.state_dict()
                ckpt.scheduler = scheduler.state_dict()
                ckpt.logfilename = logfilename
                ckpt.step = epoch + 1
                # torch.save(model.state_dict(), os.path.join(args.root, 'model_params.pt'))

    # args.mode = 'eval'
    # eval(args, model)
    # Record the overall elapsed time
    overall_time = time.time() - start_time

    # Save to CSV
    # Prepare a CSV file to log the overall time
    csv_file = os.path.join(args.root, "overall_time.csv")
    fields = ["Overall Time (seconds)"]
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)  # Write the header
        writer.writerow([overall_time])  # Write the overall time

    print(f"Overall time saved to {csv_file}: {overall_time:.2f} seconds for {args.num_epochs} epochs")


def gen_evalset(args):
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = select_dataset(args,split="test")
    eval_loader = torch.utils.data.DataLoader(eval_ds,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False, num_workers=0)

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        batches.append(img_to_task(
            x, max_num_points=args.max_num_points,
            t_noise=args.t_noise)
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'celeba')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = 'no_noise.tar' if args.t_noise is None else \
        f'{args.t_noise}.tar'
    torch.save(batches, osp.join(path, filename))
    return eval_loader


def eval(args, model, load_model=True):
    if args.mode == 'eval':
        if load_model:
            ckpt = torch.load(os.path.join(args.root, 'model_params.pt'),
                              map_location='cuda' if torch.cuda.is_available() else "cpu")
            model.load_state_dict(ckpt)
        if args.eval_logfile is None:
            eval_logfile = f'eval'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, args.data_name)
    if not osp.isdir(path):
        os.makedirs(path)

    eval_ds = select_dataset(args, split="test")
    eval_loader = torch.utils.data.DataLoader(eval_ds,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False, num_workers=0)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ravg = RunningAverage()
    model.eval()


    with torch.no_grad():
        for (x, _) in tqdm(eval_loader, ascii=True):
            x = x.to(device)
            if args.model_name == 'ConvNP':
                batch = convnp_batch_preprocessing(x, args.max_num_points)
            else:
                batch = img_to_task(x,
                                max_num_points=args.max_num_points, model_name=args.model_name)

            outs = model(batch, args.eval_num_samples)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model_name}:{args.expid} '
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
    return line, df['tar_ll'].values[0]


def plot(args, model, half_context=None):
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    if args.mode == 'plot':
        ckpt = torch.load(os.path.join(args.root, 'model_params.pt'),
                          map_location=device)
        model.load_state_dict(ckpt)
        print("load =true")

    #eval_ds = CelebA(train=False)
    eval_ds = select_dataset(args, split="val")
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    rand_ids = torch.randperm(len(eval_ds))[:1]
    # rand_ids = [4]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).to(device)
    # half_context = 'right'
    if half_context is None:
        batch = img_to_task(test_data, max_num_points=None, num_ctx=100 if test_data[0].shape[-2]<100 else 500, target_all=True)
    else:
        batch = half_img_to_task(test_data, leftright_updown=half_context, target_all=True)

    model.eval()
    with torch.no_grad():
        if args.model_name in ["np", "anp", "bnp", "banp", "tnpa"]:
            outs = model.predict(batch.xc, batch.yc, batch.x_coord, num_samples=args.eval_num_samples)
        else:
            outs = model.predict(batch.xc, batch.yc, batch.x_coord, num_samples=50)
            # outs = model.predict(batch.xc, batch.yc, batch.x_coord, num_samples=args.eval_num_samples)
            # outs = model(batch)

    ll = outs.log_prob(batch.y_coord).squeeze().sum(-1)/1024
    temp = ll.mean()

    def map_to_indices(u, H):
        # map from (-1, 1) to (0, H)
        x = (u+1)*(H-1)/2
        return x
    mu = outs.mean
    sigma = outs.scale

    # Plot context points
    img_H = min(test_data.shape[-1], 100)
    blue_img = tf.tile(tf.constant([[[0., 0., 1.]]]), [img_H, img_H, 1]) # blue rgb [0, 0, 1]
    indices = tf.cast(map_to_indices(batch.xc[0].cpu(), img_H), tf.int32) # change coordinates from (-1, 1) to (0, H)
    if batch.yc.shape[-1] == 1:
        updates = tf.tile(batch.yc[0]+0.5, [1, 3]) # from (-0.5, 0.5) to (0,1)
    else:
        updates = batch.yc[0]+0.5
    # print(updates.device)
    # print(blue_img.device)
    # print(indices.device)
    context_img = tf.tensor_scatter_nd_update(blue_img.cpu(), indices.cpu(), updates.cpu())

    # Show images from the first sample in the batch
    n_rows =4
    fig, axs = plt.subplots((n_rows+2)//2, 4, figsize=(8, 8), dpi=300) # first row to show mask and original image

    axs[0, 0].imshow(context_img.numpy())
    axs[0, 0].axis('off')
    axs[0, 0].set_title(f'{batch.xc.shape[1]} context points')

    if batch.y_coord.shape[-1] == 1:
        axs[0, 1].imshow(batch.y_coord[0].reshape([img_H, img_H, -1]).repeat([1, 1, 3]).cpu().numpy()+0.5)
    else:
        axs[0, 1].imshow(batch.y_coord[0].reshape([img_H, img_H, -1]).cpu().numpy() + 0.5)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('original image')


    mean_mu = torch.mean(mu, dim=0, keepdim=True)
    mean_std = torch.mean(sigma, dim=0, keepdim=True)
    var_coeff = 2
    if mean_mu.shape[-1] == 1: # gray scale
        mean = tf.tile(tf.reshape(mean_mu, (img_H, img_H, 1)), [1, 1, 3])
        var = tf.tile(tf.reshape(mean_std, (img_H, img_H, 1)), [1, 1, 3])
        var_coeff = 1
    else:
        mean = tf.reshape(mean_mu.cpu(), (img_H, img_H, -1))
        var =var_coeff * tf.reshape(mean_std.cpu(), (img_H, img_H, -1))
    axs[0, 2].imshow(mean.numpy() + 0.5, vmin=0., vmax=1.)
    axs[0,3].imshow(var.numpy(), vmin=0., vmax=1.)
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')
    axs[0, 2].set_title(' Avg Predicted mean')
    axs[0, 3].set_title(' Avg Predicted %.1f variance'%var_coeff)

    # plt.show()
    # Plot mean and variance
    temp = axs.shape
    for i in range(n_rows):
        row_idx = i//2 +1
        col_idx = 2*(i%2)
        if mu.shape[-1]== 1:
            mean = tf.tile(tf.reshape(mu[i], (img_H, img_H, 1)), [1, 1, 3])
            var = tf.tile(tf.reshape(sigma[i], (img_H, img_H, 1)), [1, 1, 3])
        else:
            mean = tf.reshape(mu[i].cpu(), (img_H, img_H, -1))
            var = var_coeff*tf.reshape(sigma[i].cpu(), (img_H, img_H, -1))
        axs[row_idx, col_idx].imshow(mean.numpy()+0.5, vmin=0., vmax=1.)
        axs[row_idx, col_idx+1].imshow(var.numpy(), vmin=0., vmax=1.)
        axs[row_idx, col_idx].axis('off')
        axs[row_idx, col_idx+1].axis('off')
        axs[row_idx, col_idx].set_title('z sample %d \n Predicted mean'%(i+1))
        axs[row_idx, col_idx+1].set_title('z sample %d \n Predicted variance x %.1f'%(i+1, var_coeff))
    plt.tight_layout()
    plt.savefig("regression/results/plots/%s_%s.png"%(args.data_name, args.model_name))
    plt.show()
    print("")


if __name__ == '__main__':
    main()
