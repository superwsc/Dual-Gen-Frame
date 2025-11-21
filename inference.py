import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from networks.DGF_64_1LK0508 import DGF
from postnetwork.UNet_mod import UNet

from dataloaders.data_rgb import get_validation_data
import utils
from skimage import img_as_ubyte
from config import Config 
opt = Config('training_BLGM.yml')

parser = argparse.ArgumentParser(description='Image Enhancement')

parser.add_argument('--input_dir', default='RLE/evaluation/', type=str, help='Directory of validation images')
parser.add_argument('--restore_dir', default='RLE/evaluation/ours/restore_lowextremelydark_ours', type=str, help='Directory for results')
parser.add_argument('--post_dir', default='RLE/evaluation/ours/post_lowextremelydark_ours', type=str, help='Directory for results')

parser.add_argument('--weights', default='checkpoints_IGLFM/model_IGLFM.pth', type=str, help='Path to weights')
parser.add_argument('--post_weights', default='checkpoints_BLGM/model_BLGM.pth', type=str, help='Path to weights')

parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save Enahnced images in the result directory')
parser.add_argument('--post', action='store_true', help='Save post images in the result directory')
parser.add_argument('--restore', action='store_true', help='Save restore images in the result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.restore:
    utils.mkdir(args.restore_dir)
    model_restoration = DGF(device = 'cuda:1')
    utils.load_checkpoint(model_restoration,args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration=nn.DataParallel(model_restoration)
    model_restoration.eval()
if args.post:
    utils.mkdir(args.post_dir)
    model_post = UNet(n_channels = 32)
    utils.load_checkpoint(model_post,args.post_weights)
    print("===>Testing using post_weights: ", args.post_weights)
    model_post.cuda()
    model_post=nn.DataParallel(model_post)
    model_post.eval()


img_options_val = {'patch_size':opt.TRAINING.VAL_PS}
test_dataset = get_validation_data(args.input_dir, img_options_val)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)


def cal_hist(input_tensor):
    
    bins = 64  
    min_val = 0.0  
    max_val = 1.0  

    batch_size = input_tensor.size(0)
    histograms = torch.zeros(batch_size, bins).cuda()

    
    for i in range(batch_size):
        
        image = input_tensor[i]
        gray_image = torch.mean(image, dim=0, keepdim=False)
        
        hist = torch.histc(image, bins=bins, min=min_val, max=max_val)

        histograms[i] = hist / 65536.0  

    return histograms
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]

        if args.restore:
            rgb_restored, temp_feat = model_restoration(rgb_noisy)  
        else:
            rgb_restored, temp_feat = rgb_noisy  

        if args.post:
            input_hist = cal_hist(rgb_noisy)
            post_restored = model_post(temp_feat, input_hist)

            rgb_restored = torch.clamp(rgb_restored,0,1)
            post_restored = torch.clamp(post_restored,0,1) 
            psnr_val_rgb.append(utils.batch_PSNR(post_restored, rgb_gt, 1.))
        else:
            rgb_restored = torch.clamp(rgb_restored,0,1)
            psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))

        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                if args.restore:
                    
                    rgb_restored_img = img_as_ubyte(rgb_restored[batch])
                    
                    utils.save_img(args.restore_dir +'/'+ filenames[batch][:-4] + '.png', rgb_restored_img)

                if args.post:
                    post_restored = post_restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                    post_restored_img = img_as_ubyte(post_restored[batch])
                    utils.save_img(args.post_dir +'/'+ filenames[batch][:-4] + '.png', post_restored_img)     
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb))
