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
from lossfuncs.loss import *
from myutils.mask_functions import pixels2mask, mask2pixels
from networks.imageunet import init_network
# from myutils.common_util import *
# from layers.scheduler import *
from datasets.datastes import ImageData,MyBalanceClassSampler, MyBalanceClassSampler2
# from myutils.cal_util import AverageMeter, metric
from myutils.augumentation_util import dataAugumentation,dataAugumentation3
from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import warnings
import random
import segmentation_models_pytorch as smp
import shutil

path='E:/data_set/steel/'
train_csv = pd.read_csv(path + 'train.csv')
train_csv['ImageId'], train_csv['ClassId'] = zip(*train_csv['ImageId_ClassId'].str.split('_'))
train_csv['ClassId'] = train_csv['ClassId'].astype(int)
train_csv = pd.pivot(train_csv, index='ImageId', columns='ClassId', values='EncodedPixels')
train_csv['defects'] = train_csv.count(axis=1)

train_data, val_data = train_test_split(train_csv, test_size=0.2, stratify=train_csv['defects'], random_state=69)
train_dataset = ImageData(train_data, path, dataAugumentation3, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                          subset="train")
# train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
myTrainSample = MyBalanceClassSampler(train_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=12, sampler=myTrainSample, pin_memory=True)

valid_dataset = ImageData(val_data, path, augmentation=dataAugumentation3, mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225), subset="valid")
# valid_loader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=True)
myValSample = MyBalanceClassSampler(valid_dataset)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, sampler=myValSample, pin_memory=True)



def mask2rle(img):
    '''+
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if not pd.isnull(label):
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos-1:(pos + le-1)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')


    return fname, masks
"""输入tensor类型的（batchsize，4，h，w）大小的mask，输出tensor类型的（batchsize，5，h，w）的mask"""
def mask4tomask5(mask):
    batch_size,_,height,width=mask.shape
    mask=np.copy(mask)
    mask5=np.zeros([batch_size,5,height,width])
    mask5[:,:-1,:,:]=mask
    for i in range(batch_size):
        logi=mask[i,0,:,:]
        for ie in range(1, 4):
            logi = np.logical_or(logi, mask[i,ie,:,:])
        mask5[i,4,:,:]=np.logical_not(logi).astype(np.float32)

    mask5=torch.from_numpy(mask5)
    return mask5



"""应该是要改成取最大值，然后把第5维去掉"""
# def predict(X):
#     X_p= X >= (torch.max(X, 1)[0].unsqueeze(1))
#     X_p=X_p[:,:-1,:,:]
#     preds= np.copy(X_p).astype('uint8')
#     return preds

def predict(X,):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > 0.5).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos




    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        dice, dice_neg, dice_pos, _, _ = metric(outputs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(outputs)
        iou = compute_iou_batch(preds, targets, classes=[1])               ##??
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    # pred[label == ignore_index] = 0   ##??
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    labels = np.copy(labels) # copy is imp
    preds = np.array(outputs) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou





class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 12, "val": 32}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 1e-4
        self.num_epochs = 60
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([3,5,1,2])).cuda()
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        # self.criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,momentum=0.9, weight_decay=0.0001)
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=0.0001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=2, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            "train":train_loader,
            "val":valid_loader
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        output=outputs.permute(0,3,1,2).reshape(-1,4)
        masks=masks.permute(0,3,1,2).reshape(-1,4)
        loss = self.criterion(output, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            # loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss ) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state,"./record.pth")
                torch.save(self.net.state_dict(), "E:/pycharm_project/steel/code/densebalanceseg.pth")
                # torch.save(self.net,"./model")
            print()

def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
    plt.legend();
    plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 100
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # model=torch.load("./model")
    """5类，包括没有的"""
    model = smp.Unet("efficientnet-b1", encoder_weights='imagenet', classes=4, activation=None)
    # model.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/b5fpnbalanceseg.pth"))
    sample_submission_path = 'E:/pycharm_project/STEEL/sample_submission.csv'
    train_df_path = 'E:/data_set/steel/train.csv'
    data_folder = "E:/data_set/steel/"
    test_data_folder = "E:/data_set/steel/test_images"

    model_trainer = Trainer(model)
    # sss = torch.load("./record.pth")
    # model_trainer.best_loss=sss['best_loss']
    # model_trainer.optimizer.load_state_dict(sss["optimizer"])
    # model_trainer.optimizer = optim.Adam(model_trainer.net.parameters(), lr=5e-5)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores # overall dice
    iou_scores = model_trainer.iou_scores



    plot(losses, "BCE loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")