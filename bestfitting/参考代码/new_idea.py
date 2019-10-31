import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm_notebook
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, sampler

class SteelDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx].name
        target = self.df.iloc[idx][2]
        target=target.strip('[').strip(']').strip('\n').split(" ")
        target=map(float,target)
        label=np.zeros([4])
        for idx,i in enumerate(target):
            label[idx]=i
        img=self.df.iloc[idx][1]
        img = img.strip('[').strip(']').strip('\n').split(" ")
        input=np.zeros([16])
        index=0
        for _,i in enumerate(img):
            if i!='':
                input[index]=i
                index+=1
        input=input.reshape((1,4,4))
        label=label.reshape(1,1,4)
        return input,label

    def __len__(self):
        return len(self.fnames)


data=pd.read_csv('E:/pycharm_project/steel/code/submit/floatcls.csv')
dataset=SteelDataset(data)
dataloader=DataLoader(dataset,batch_size=12568,num_workers=0,pin_memory=True,shuffle=True)
m=torch.nn.Conv2d(1,1,[4,1],padding=0,bias=False)
m.weight=torch.nn.Parameter(torch.tensor([[[[0.3],[0.1],[0.1],[0.5]]]]))
m.cuda()
criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(m.parameters(),lr=1e-2)
for epoch in range(100):
    epoch_loss=0
    for itr,batch in enumerate(dataloader):
        images,targets=batch
        optimizer.zero_grad()
        images=images.float().cuda()
        targets=targets.float().cuda()
        output=torch.sigmoid(m(images))
        loss=criterion(output,targets)
        loss.backward()
        optimizer.step()
        loss.item()
        epoch_loss += loss.item() * 12568/12568
    print(epoch_loss)
    print(m.weight)
