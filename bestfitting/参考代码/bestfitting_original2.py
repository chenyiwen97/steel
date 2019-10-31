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
import albumentations as albu
from sklearn.model_selection import train_test_split
import time
from lossfuncs.loss import *

# path = 'E:/data_set/steel/'
# path = 'D:/chenyiwen/steel/steel/'
path = 'H:/input/severstal-steel-defect-detection/'

train_csv = pd.read_csv(path + 'train.csv')
train_csv['ImageId'], train_csv['ClassId'] = zip(*train_csv['ImageId_ClassId'].str.split('_'))
train_csv['ClassId'] = train_csv['ClassId'].astype(int)
train_csv = pd.pivot(train_csv, index = 'ImageId', columns = 'ClassId', values = 'EncodedPixels')
train_csv['defects'] = train_csv.count(axis=1)

def mask2pixels(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pixels2mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    # print(fname)
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

def dataAugumentation(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = []
    if phase == "train":
        trans.extend(
            [
                albu.RandomCrop(256, 256, always_apply = True),
                albu.HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    trans.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
        ]
    )
    all_trans = albu.Compose(trans)
    return all_trans

class ImageData(Dataset):
    def __init__(self, df, path, augmentation, mean, std, subset="train"):
        super().__init__()
        self.df = df
        self.augmentation = augmentation(subset, mean, std)
        self.subset = subset
        
        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
    
    @staticmethod
    def numpy_to_torch(s):
        return torch.from_numpy(s.astype(np.float32))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):     
        imgId, mask = pixels2mask(idx, self.df)
        imgPath = self.data_path + imgId
        # print(imgPath)
        # img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(imgPath)                 
        augmented = self.augmentation(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask']
        img = np.transpose(img, [2, 0, 1])
        # img = np.expand_dims(img, axis=0)
        mask = np.transpose(mask, [2, 0, 1])
        # mask = np.expand_dims(mask, axis=0)  
        # mask = mask[0].permute(2,0,1) # 1x4x256x1600
        x = self.numpy_to_torch(img)
        y = self.numpy_to_torch(mask)
        return x, y

# train_data, val_data = train_test_split(train_csv, test_size = 0.2, stratify = train_csv['defects'], random_state=69)
train_data = train_csv

train_set = ImageData(train_data, path, dataAugumentation, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), subset="train")

train_set_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)

def metric(logit, truth, threshold=0.5):
    dice = dice_score(logit, truth, threshold=threshold)
    return dice

device = torch.device("cuda:0")

from networks.imageunet import init_network
model_params = {}
model_params['architecture'] = "unet_resnet34_cbam_v0a"
net = init_network(model_params)

# net.load_state_dict(torch.load("D:/chenyiwen/steel/george/fcn/model_bceabdnormalization.pth"))
# net.eval()

net = net.to(device)

# criterion = ComboLoss({'dice': 1.0, 'bce': 1.0}, per_image=True).cuda()
criterion = nn.BCEWithLogitsLoss().cuda()
# criterion = SymmetricLovaszLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.00001)
# optimizer = torch.optim.SGD(net.parameters(), weight_decay=1e-4, lr = 0.001, momentum=0.9)

# summary(net, (1, 32, 32))
phase = "train"
losses = []
### ========== training phrase ========== ###  
for epoch in range(50):
    start = time.strftime("%H:%M:%S")
    print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")      
    net.train()         
    for ii, batch in enumerate(train_set_loader):
        data, target = batch
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()          

        probs = F.sigmoid(output)
        dice = metric(probs, target)

    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))
    print('Epoch: {} - dice: {:.6f}'.format(epoch + 1, dice.item()))
    losses.append(loss)

def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores)), scores, label=f'train {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}')
    plt.legend()
    plt.show()

plot(losses, "all loss")

torch.save(net.state_dict(), "./model_withhoriflip.pth")