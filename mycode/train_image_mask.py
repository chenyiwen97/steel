import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
import albumentations as albu
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
#生成训练集的mask以供观察
def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 5), dtype=np.float32) # float32 is V.Imp
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
    logi=masks[:,:,0]
    for ie in range(1,4):
        logi=np.logical_or(logi,masks[:, :, ie])
        # print(logi.sum())
    masks[:, :, 4]=np.logical_not(logi).astype(np.float32)

    return fname, masks

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
source_path="E:/data_set/steel/train_images"
des_path="E:/data_set/steel/train_image_mask/"
train_df_path = 'E:/data_set/steel/train.csv'
df = pd.read_csv(train_df_path)
df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
df['defects'] = df.count(axis=1)
for idx in range(len(df['defects'])):
    image_id, mask = make_mask(idx,df)
    mask=mask.astype(np.uint8)
    image_path = os.path.join(source_path, image_id)
    img = cv2.imread(image_path)
    # fig, ax = plt.subplots(figsize=(15, 15))
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    # ax.imshow(img)
    # plt.show()
    cv2.imwrite(des_path+image_id,img)