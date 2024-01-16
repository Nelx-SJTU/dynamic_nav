from torch import nn
import torch.nn.functional as F
# from spatial_correlation_sampler import spatial_correlation_sample
import torch as t


def multi_l1_loss(output,target):
    def one_scale(output, target):
        b, _, h, w = output.size()
        target_scaled = F.interpolate(target, (h, w), mode='area')
        loss_ = 0
        loss_ += F.l1_loss(output, target_scaled)
        return loss_
    weights = [0.32, 0.08, 0.02, 0.01]
    loss = 0
    for out, weight in zip(output, weights):
        loss += weight * one_scale(out, target)
    return loss
    



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,padding = None):
    if padding == None:
        padding = (kernel_size-1)//2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes, kernel_size=4):  #0=s(i-1)-2p+k
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1) #(batchsize, 441, H,W)
    return F.leaky_relu_(out_corr, 0.1)

def adjust_learning_rate(initial_lr,lr_decay_step,episode,optimizer):
    if lr_decay_step > 0:
        learning_rate = 0.9 * initial_lr * (
                lr_decay_step - episode) / lr_decay_step + 0.1 * initial_lr
        if episode > lr_decay_step:
            learning_rate = 0.1 * initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        learning_rate = initial_lr
    return learning_rate

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    loss = 0
    weight = 1.
    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
    return loss

def ssim_loss(x, y):
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return t.clamp((1 - ssim) / 2, 0, 1).mean()