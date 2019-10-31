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
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=5, activation=None)
# model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientseg.pth'))
model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/unetb0resize.pth'))
# model = smp.FPN("efficientnet-b5", encoder_weights="imagenet", classes=5, activation=None)
# model.load_state_dict(torch.load('./b5cropfpnseg.pth'))
sample_submission_path = 'E:/data_set/steel/train.csv'
test_data_folder = "E:/data_set/steel/resize_image/"
model.eval()
model.cuda()
test_data=pd.read_csv(sample_submission_path)
for index in range(0, len(test_data),4):
    pic, _ = test_data['ImageId_ClassId'][index].split('_')
    print(pic)
    img = cv2.imread(test_data_folder + pic)
    img2=cv2.imread('E:/data_set/steel/train_image_mask/'+pic)
    sget_transforms = get_transforms('val')
    aug = sget_transforms(image=img)
    image = aug['image'].unsqueeze(0)
    image = image.cuda()
    output = model(image).data.cpu()
    pred=predict(output)

    for ch in range(4):
        contours, _ = cv2.findContours(pred[0,ch,:, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1, 2,2)
    plt.imshow(img2)
    plt.show()
    # cv2.imshow(image[1])
    # for i in range(4):
    #     to_submit = post_process(pred[0, i, :, :], 0.5, 800)
    #     # to_submit=pred[0, i, :, :]
    #     submit=mask2rle(to_submit)
    #     test_data['EncodedPixels'][index+i]=submit


# test_data.to_csv('E:/pycharm_project/steel/code/submit/efficientsoftmaxseg.csv',index=False)