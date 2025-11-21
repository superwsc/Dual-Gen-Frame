import os
import sys
from config import Config 
opt = Config('training_IGLFM.yml')


print("batch size: " , opt.OPTIM.BATCH_SIZE)
print("patch size ", opt.TRAINING.TRAIN_PS)

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from natsort import natsorted
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

from networks.DGF_64_1LK0508 import DGF
from utils.losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from networks.Discriminator import Discriminator
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results')
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, 'models')

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir_RLE = opt.TRAINING.TRAIN_DIR_RLE
#print(train_dir_RLE)
train_dir_KCL = opt.TRAINING.TRAIN_DIR_KCL
val_dir_KCL   = opt.TRAINING.VAL_DIR_KCL
val_dir_RLE   = opt.TRAINING.VAL_DIR_RLE
save_images = opt.TRAINING.SAVE_IMAGES

######### Model ###########
model_restoration = DGF(device = 'cuda:1')
model_restoration.cuda()

model_discriminator = Discriminator()
model_discriminator.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
optimizer_d = optim.Adam(model_discriminator.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_best.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

if len(device_ids)>1:
    model_discriminator = nn.DataParallel(model_discriminator, device_ids = device_ids)
######### Loss ###########
##L2 loss
criterion = CharbonnierLoss().cuda()

######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}
train_dataset_KCL = get_training_data(train_dir_KCL, img_options_train)
train_dataset_RLE = get_training_data(train_dir_RLE, img_options_train)
combined_train_dataset = ConcatDataset([train_dataset_KCL, train_dataset_RLE])
train_loader = DataLoader(dataset=combined_train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)
#train_loader = DataLoader(dataset=train_dataset_RLE, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

val_dataset_KCL = get_validation_data(val_dir_KCL, img_options_train)
val_dataset_RLE = get_validation_data(val_dir_RLE, img_options_train)
combined_val_dataset = ConcatDataset([val_dataset_KCL, val_dataset_RLE])
val_loader = DataLoader(dataset=combined_val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//2 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")



for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0):    

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        #data[0]->gt    data[1]->ll
        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch>5:
            target, input_ = mixup.aug(target, input_)

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)  
        #L1 loss
        loss = criterion(restored, target)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0:
            if save_images:
                utils.mkdir(result_dir + '%d/%d'%(epoch,i))
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1) 
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                    if save_images:
                        target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                        
                        for batch in range(input_.shape[0]):

                            temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255),axis=1)
                            if np.isnan(temp).any():
                                print("temp is NAN")
                                sys.exit()
                            

                            utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))

                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
            
            model_restoration.train()

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

