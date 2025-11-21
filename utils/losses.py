import torch
import torch.nn as nn
import torch.nn.functional as F

def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)

    if torch.sum(union) != 0:
        iou = torch.sum(intersection) / torch.sum(union)
    else:
        iou = 1

    return iou

def calculate_miou(mask1, mask2):
    # 计算每个类别的 IoU
    iou_foreground = calculate_iou(mask1, mask2)
    iou_background = calculate_iou(1 - mask1, 1 - mask2)  # 对于背景，需要将 mask 取反

    # 计算 mIoU（平均 IoU）
    miou = (iou_foreground + iou_background) / 2.0
    return miou

def calculate_dice_coefficient(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    #intersection = mask1 * mask2

    #mask 范围是0和255，但是intersection是0和1
    dice_coefficient = (2.0 * torch.sum(intersection)) / (torch.sum(mask1) + torch.sum(mask2))
    return dice_coefficient


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class SegLoss(nn.Module):
    '''
    mIoU与Dice二者0.5加权平均
    '''
    def __init__(self):
        super(SegLoss, self).__init__()
        

    def forward(self, x, y):
        miou = calculate_miou(x, y)
        #miou_loss = 1 - miou

        dice = calculate_dice_coefficient(x, y)
        #dice_loss = 1 - dice
        loss = 0.5 * miou + 0.5 * dice
        #loss =  miou

        #loss = 1 - loss
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

