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
# import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
import albumentations as albu
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch.transforms import ToTensor
# import UNet_starter_kernel as un

def get_transforms(phase):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now

            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
def get_transforms2(phase):
    list_transforms = []
    list_transforms.extend(
        [
            HorizontalFlip(p=1),
            Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
def predict(X):
    X_p= X >= (torch.max(X, 1)[0].unsqueeze(1))
    X_p=X_p[:,:-1,:,:]
    preds= np.copy(X_p).astype('uint8')
    return preds
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
def post_process(probability, threshold = 0.5, min_size = 200):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return predictions

model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=5, activation=None)
model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientseg.pth'))
# model = smp.Unet("se_resnext50_32x4d", encoder_weights="imagenet", classes=5, activation=None)
# model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/seunetseg.pth'))
# model = smp.FPN("efficientnet-b5", encoder_weights="imagenet", classes=5, activation=None)
# model.load_state_dict(torch.load('./b5cropfpnseg.pth'))
sample_submission_path = 'E:/data_set/steel/sample_submission.csv'
test_data_folder = "E:/data_set/steel/test_images/"
model.eval()
model.cuda()
test_data=pd.read_csv(sample_submission_path)
sget_transforms = get_transforms('val')
size_threshold=[300,800,800,800]
for index in range(0, len(test_data),4):
    pic, _ = test_data['ImageId_ClassId'][index].split('_')
    img = cv2.imread(test_data_folder + pic)

    aug = sget_transforms(image=img)
    img = aug['image'].unsqueeze(0)
    img = img.cuda()
    img2=torch.flip(img,[3])
    # print(img[0,0,0,0])
    # print(img2[0,0,0,-1])
    output = model(img).data.cpu()
    output2=model(img2).data.cpu()
    output2=torch.flip(output2,[3])
    output=output+output2
    pred=predict(output)
    # cv2.imshow(image[1])
    for i in range(4):
        to_submit = post_process(pred[0, i, :, :], 0.5, size_threshold[i])
        # to_submit=pred[0, i, :, :]
        submit=mask2rle(to_submit)
        test_data['EncodedPixels'][index+i]=submit


test_data.to_csv('E:/pycharm_project/steel/code/submit/efficientsoftmaxseg.csv',index=False)