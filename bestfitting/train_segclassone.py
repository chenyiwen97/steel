import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2
from PIL import Image
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
from networks.imageunet import init_network
from myutils.common_util import *
from layers.scheduler import *
from myutils.cal_util import AverageMeter, metric
import shutil

### ============== some utils function =========== ####




def main():

    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    #### =========== 定义模型 (主要包括网络结构， loss函数，还有优化器设置) =========== ####
    model_params = {}
    model_params['architecture'] = arch
    model = init_network(model_params)
    # move network to gpu
    model.cuda()
    # define loss function (criterion)
    try:
        criterion = eval(lossfnc)().cuda()
    except:
        raise(RuntimeError("Loss {} not available!".format(lossfnc)))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

    start_epoch = 0
    best_epoch = 0
    best_dice = 0

    # define scheduler -- 动态调整学习率
    try:
        scheduler = eval(scheduler2)()
    except:
        raise (RuntimeError("Scheduler {} not available!".format(scheduler2)))
    optimizer = scheduler.schedule(model, start_epoch, epochs)[0]

    # Data loading code
    train_csv = pd.read_csv(path + 'train.csv') # 50272
    # 减少样本数量 -- 样本数量减少十倍
    train_csv = train_csv[:7000]
    train_csv['ImageId'], train_csv['ClassId'] = zip(*train_csv['ImageId_ClassId'].str.split('_'))
    train_csv['ClassId'] = train_csv['ClassId'].astype(int)
    train_csv = pd.pivot(train_csv, index = 'ImageId', columns = 'ClassId', values = 'EncodedPixels')
    train_csv['defects'] = train_csv.count(axis=1)

    train_data, val_data = train_test_split(train_csv, test_size = 0.2, stratify = train_csv['defects'], random_state=69)
    train_dataset = ImageDataWithValidCrop(train_data, path, dataAugumentationAfterCrop, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), subset="train")
    # train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    myTrainSample = MyBalanceClassSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, sampler=myTrainSample, pin_memory=True)

    valid_dataset = ImageDataWithValidCrop(val_data, path, augmentation = dataAugumentationAfterCrop, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), subset="train")
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=True)
    myValSample = MyBalanceClassSampler(valid_dataset)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=8, sampler=myValSample, pin_memory=True)

    start_epoch += 1
    for epoch in range(start_epoch, epochs + 1):

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        lr_list = scheduler.step(model, epoch, epochs)
        lr = lr_list[0]

        # train for one epoch on train set
        iter, train_loss, train_dice = train(train_loader, model, criterion, optimizer, epoch, lr=lr)

        with torch.no_grad():
                valid_loss, valid_dice = validate(valid_loader, model, criterion, epoch)

        train_loss_list.append(train_loss)
        train_dice_list.append(train_dice)
        val_loss_list.append(valid_loss)
        val_dice_list.append(valid_dice)

        # remember best loss and save checkpoint
        is_best = valid_dice >= best_dice
        if epoch > 10:
            if is_best or epoch == epochs:
                best_epoch = epoch
                best_dice = valid_dice
                print('\r', end='', flush=True)
                model_name = 'epoch' + '%03d' % epoch + '_' + '%.2f' % best_dice
                save_model(model, model_out_dir, epoch, model_name, optimizer=optimizer,
                                best_epoch=best_epoch, best_dice=best_dice)


        myplot(train_loss_list, "train_loss")
        myplot(train_dice_list, "train_dice")
        myplot(val_loss_list, "val_loss")
        myplot(val_dice_list, "val_dice")

def train(train_loader, model, criterion, optimizer, epoch, lr=1e-5):
    start = time.strftime("%H:%M:%S")
    print(f"Starting epoch: {epoch} | phase: Train | ⏰: {start}")   
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()
    # switch to train mode
    model.train()
    num_its = len(train_loader)
    end = time.time()
    iter = 0
    print_freq = 1
    # start = time.strftime("%H:%M:%S")
    # print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")   
    for iter, iter_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images, masks = iter_data
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        if lossfnc == "SymmetricLovaszLoss":
            loss = criterion(outputs, masks, epoch=epoch)
        else:
            loss = criterion(outputs, masks)
        
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        probs = F.sigmoid(outputs)
        dice = metric(probs, masks)
        dices.update(dice.item())
        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == num_its:
            print('\r%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  | ... ' % \
                  (epoch - 1 + (iter + 1) / num_its, iter + 1, lr, losses.avg, dices.avg), \
                  end='', flush=True)
    return iter, losses.avg, dices.avg

def validate(valid_loader, model, criterion, epoch):
    start = time.strftime("%H:%M:%S")
    print("\n")
    print(f"Starting epoch: {epoch} | phase: Valid | ⏰: {start}")   
    batch_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()

    num_its = len(valid_loader)
    print_freq = 1
    for iter, iter_data in enumerate(valid_loader, 0):
        images, masks = iter_data
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        if lossfnc == "SymmetricLovaszLoss":
            loss = criterion(outputs, masks, epoch=epoch)
        else:
            loss = criterion(outputs, masks)
        probs = F.sigmoid(outputs)
        dice = metric(probs, masks)

        losses.update(loss.item())
        dices.update(dice.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == num_its:
            print('\r%5.1f   %5d  |  %0.4f  %0.4f  | ... ' % \
                  (epoch - 1 + (iter + 1) / num_its, iter + 1, losses.avg, dices.avg), \
                  end='', flush=True)
    return losses.avg, dices.avg

########### ========= 保存最佳模型 ========== ###########
def save_model(model, model_out_dir, epoch, model_name,
               optimizer=None, best_epoch=None, best_dice=None):
    
    state_dict = model.state_dict()

    model_fpath = opj(model_out_dir, '%s.pth' % model_name)
    torch.save({
        'state_dict': state_dict,
        'best_epoch': best_epoch,
        'epoch': epoch,
        'best_dice': best_dice,
    }, model_fpath)

    optim_fpath = opj(model_out_dir, '%s_optim.pth' % model_name)
    if optimizer is not None:
        torch.save({
            'optimizer': optimizer.state_dict(),
        }, optim_fpath)

def myplot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores)), scores, label=f'{name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}')
    plt.legend()
    # plt.show()
    strFile = os.path.join(model_out_dir, name) + ".png"
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile)
    print(("\n====== save " + name + ".png ====== successfully"))

########## ============ ============= #########
#################  参数的设置  #################
########## ============ ============= #########
# path = 'E:/data_set/steel/'
path = 'D:/chenyiwen/steel/steel/'
# path = 'H:/input/severstal-steel-defect-detection/'
# arch = "unet_resnet34_cbam_v0a"
arch = "unet_se_resnext50_cbam_v0a"

lossfnc = "nn.BCEWithLogitsLoss" # default is nn.BCEWithLogitsLoss()
# lossfnc = "SymmetricLovaszLoss"
# lossfnc = "DiceLoss"
# lossfnc = "BCEDiceLoss"

scheduler2 = 'Adam3'
resume = None # restore 的flag -- 暂时未使用到
epochs = 35
device = torch.device("cuda:0")
model_out_dir = "./model/1002WithValidCrop"

train_loss_list = []
train_dice_list = []
val_loss_list = []
val_dice_list = []

if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))

    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    main()

    myplot(train_loss_list, "train_loss")
    myplot(train_dice_list, "train_dice")
    myplot(val_loss_list, "val_loss")
    myplot(val_dice_list, "val_dice")

    print('\nsuccess!')