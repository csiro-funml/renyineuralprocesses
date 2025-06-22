import torch
from addict import Dict
import einops
from torch.distributions import StudentT
import torch.nn.functional as F

def img_to_task(img, num_ctx=None,
        max_num_points=None, target_all=False, t_noise=None, model_name=None, noise_strategy=None):
    B, C, H, W = img.shape

    raw_img = img.clone()
    if H >32:
        # Define the transform to resize the image
        img = F.interpolate(img, size=(100, 100), mode='bilinear',
                                             align_corners=False)
        B, C, H, W = img.shape
        # print("+++++++++++image shape +++++++++++++", img.shape)
    img = img.view(B, C, -1)
    num_pixels = H * W


    batch = Dict()
    if max_num_points is not None:
        max_num_points = max_num_points or num_pixels
        num_ctx = num_ctx or \
                torch.randint(low=3, high=max_num_points-3, size=[1]).item()
        num_tar = max_num_points - num_ctx if target_all else \
                torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()
        num_points = num_ctx + num_tar
    else:
        max_num_points = 10
        num_ctx = num_ctx or \
                  torch.randint(low=3, high=max_num_points - 3, size=[1]).item()
        num_tar = num_pixels - num_ctx
        num_points = num_ctx + num_tar
    # print(num_ctx, num_tar)
    if torch.cuda.is_available():
        idxs = torch.FloatTensor(B, num_pixels).uniform_().argsort(-1)[...,:num_points].to(img.device)
    else:
        idxs = torch.FloatTensor(B, num_pixels).uniform_().argsort(-1)[..., :num_points].to(img.device)
    x1, x2 = idxs//W, idxs%W
    if model_name != 'ConvNP':
        batch.x = torch.stack([
            2*x1.float()/(H-1) - 1,
            2*x2.float()/(W-1) - 1], -1).to(img.device)
        batch.y = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
                .transpose(-2, -1) - 0.5).to(img.device)

        batch.xc = batch.x[:,:num_ctx]
        batch.xt = batch.x[:,num_ctx:]
        batch.yc = batch.y[:,:num_ctx]
        batch.yt = batch.y[:,num_ctx:]

        x_coord = torch.arange(W)
        y_coord = torch.arange(H)
        yy, xx = torch.meshgrid(x_coord, y_coord, indexing='ij')  # indexing='ij' ensures coordinates (y, x)

        # Flatten and stack the coordinates
        batch.x_coord = torch.stack([2*yy.flatten().float()/(H-1) - 1, 2*xx.flatten().float()/(H-1) - 1], dim=1).unsqueeze(0).repeat(B, 1, 1).to(img.device)
        batch.y_coord = (img.permute(0, 2, 1).reshape(B, -1, C) - 0.5).to(img.device)
    else: # ConvNP (grided inputs)
        mask = torch.zeros((B, H*W)).to(img.device) # (B, H, W)
        mask_xc = mask.clone()
        mask_xt = mask.clone()

        # generate a context mask based on a ratio?

        # print("number of context", num_ctx)
        # print("number of target", num_tar)
        # mask_xc[torch.arange(B)[:, None], idxs[:, :num_ctx]] = 1
        # mask_xt[torch.arange(B)[:, None], idxs[:, num_ctx:]] = 1
        # # print(torch.sum(mask_xc, dim=-1))
        # # print(torch.sum(mask_xt, dim=-1))
        # batch.xc = mask_xc.reshape(B, H, W).unsqueeze(-1).bool() #B, H, W, C
        # batch.xt = mask_xt.reshape(B, H, W).unsqueeze(-1).bool() #B, H, W, C
        # batch.yc = batch.xc * raw_img.permute(0, 2, 3, 1)
        # batch.yt = batch.xt * raw_img.permute(0, 2, 3, 1)
        # batch.x = batch.xc + batch.xt
        # batch.y = batch.yc + batch.yt
        # batch.mask_xc = mask_xc
        # batch.mask_xt = mask_xt

    if t_noise is not None:
        # print("using noise")
        if noise_strategy == 'y':
            e_y = torch.randn(batch.yc.shape) * 1
            e_y = e_y.to(batch.yc.device)
            batch.yc = (1 - t_noise) * batch.yc + t_noise * e_y
        elif noise_strategy == 'x':
            e_x = torch.randn(batch.xc.shape) * 1
            e_x = e_x.to(batch.xc.device)
            batch.xc = (1 - t_noise) * batch.xc + t_noise * e_x
        else:  # both x, y
            e_y = torch.randn(batch.yc.shape) * 1
            e_y = e_y.to(batch.yc.device)
            batch.yc = (1 - t_noise) * batch.yc + t_noise * e_y
            e_x = torch.randn(batch.xc.shape) * 1
            e_x = e_x.to(batch.xc.device)
            batch.xc = (1 - t_noise) * batch.xc + t_noise * e_x
    return batch

def coord_to_img(x, y, shape):
    x = x.cpu()
    y = y.cpu()
    B = x.shape[0]
    C, H, W = shape

    I = torch.zeros(B, 3, H, W)
    I[:,0,:,:] = 0.61
    I[:,1,:,:] = 0.55
    I[:,2,:,:] = 0.71

    x1, x2 = x[...,0], x[...,1]
    x1 = ((x1+1)*(H-1)/2).round().long()
    x2 = ((x2+1)*(W-1)/2).round().long()
    for b in range(B):
        for c in range(3):
            I[b,c,x1[b],x2[b]] = y[b,:,min(c,C-1)]

    return I

def task_to_img(xc, yc, xt, yt, shape):
    xc = xc.cpu()
    yc = yc.cpu()
    xt = xt.cpu()
    yt = yt.cpu()

    B = xc.shape[0]
    C, H, W = shape

    xc1, xc2 = xc[...,0], xc[...,1]
    xc1 = ((xc1+1)*(H-1)/2).round().long()
    xc2 = ((xc2+1)*(W-1)/2).round().long()

    xt1, xt2 = xt[...,0], xt[...,1]
    xt1 = ((xt1+1)*(H-1)/2).round().long()
    xt2 = ((xt2+1)*(W-1)/2).round().long()

    task_img = torch.zeros(B, 3, H, W).to(xc.device)
    task_img[:,2,:,:] = 1.0
    task_img[:,1,:,:] = 0.4
    for b in range(B):
        for c in range(3):
            task_img[b,c,xc1[b],xc2[b]] = yc[b,:,min(c,C-1)] + 0.5
    task_img = task_img.clamp(0, 1)

    completed_img = task_img.clone()
    for b in range(B):
        for c in range(3):
            completed_img[b,c,xt1[b],xt2[b]] = yt[b,:,min(c,C-1)] + 0.5
    completed_img = completed_img.clamp(0, 1)

    return task_img, completed_img


def half_img_to_task(img, leftright_updown= None, target_all=False, t_noise=None, model_name=None, noise_strategy=None):
    B, C, H, W = img.shape
    # print("+++++++++++image shape +++++++++++++", img.shape)
    num_pixels = H*W
    raw_img = img.clone()
    img = img.view(B, C, -1)


    batch = Dict()
    #TODO: change this
    num_ctx = num_pixels//2
    if leftright_updown == 'left':
        x_c = torch.arange(H)
        y_c = torch.arange(W//2)
        context_img = raw_img[:, :, :, :W//2]
    elif leftright_updown == 'right':
        x_c = torch.arange(H)
        y_c = torch.arange(W//2, W)
        context_img = raw_img[:, :, :, W // 2:]
    elif leftright_updown == 'up':
        x_c = torch.arange(H//2)
        y_c = torch.arange(W)
        context_img = raw_img[:, :, :H//2, :]
    elif leftright_updown == 'down':
        x_c = torch.arange(H//2, H)
        y_c = torch.arange(W)
        context_img = raw_img[:, :, H//2:, :]
    # define num_ctx, num_tar
    xxc, yyc = torch.meshgrid(x_c, y_c, indexing='ij')  # indexing='ij' ensures coordinates (y, x)
    # Flatten and stack the coordinates
    batch.xc = torch.stack([2 * xxc.flatten().float() / (H - 1) - 1, 2 * yyc.flatten().float() / (H - 1) - 1],
                                dim=1).unsqueeze(0).repeat(B, 1, 1).to(img.device)
    # y_k_res = einops.rearrange(y_k, 'b n c -> b c n 1').contiguous()
    batch.yc = einops.rearrange(context_img-0.5, 'b c w d -> b (w d) c').contiguous().to(img.device)

    ###############
    x_coord = torch.arange(W)
    y_coord = torch.arange(H)
    yy, xx = torch.meshgrid(x_coord, y_coord, indexing='ij')  # indexing='ij' ensures coordinates (y, x)

    # Flatten and stack the coordinates
    batch.x_coord = torch.stack([2*yy.flatten().float()/(H-1) - 1, 2*xx.flatten().float()/(H-1) - 1], dim=1).unsqueeze(0).repeat(B, 1, 1).to(img.device)
    batch.y_coord = (img.reshape(B, -1, C) - 0.5).to(img.device)

    return batch