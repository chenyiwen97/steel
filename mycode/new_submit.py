# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import pdb
import time
import warnings
import random
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
from albumentations.pytorch.transforms import ToTensor
from torchvision import models
from efficientnet_pytorch import EfficientNet
import os
# model = smp.Unet("resnet18", encoder_weights="imagenet", classes=5, activation="softmax")
# model.load_state_dict(torch.load("./softmaxmodel.pth")["state_dict"])
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc_features = model.fc.in_features
        self.fc = nn.Sequential(nn.Linear(self.fc_features, 32),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(32, 4)
                                    )
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model=models.resnet34(pretrained=True)
fc_features = model.fc.in_features
model.fc=nn.Linear(fc_features,4)

# model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet34fourclsWithResize.pth'))
model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet34fourcls.pth'))
model.eval()
model.cuda()

model2 = EfficientNet.from_pretrained('efficientnet-b3')
fc_features =model2._fc.in_features
model2._fc=nn.Linear(fc_features,4)
model2.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientnetfourcls.pth'))
model2.eval()
model2.cuda()

model3=models.resnet50(pretrained=True)
fc_features = model3.fc.in_features
model3.fc=nn.Linear(fc_features,4)
model3.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet50fourcls.pth'))
model3.eval()
model3.cuda()

model4=models.resnet34(pretrained=True)
model4 = Net(model4)
model4.load_state_dict(torch.load('C:/Users/PC/Documents/Tencent Files/867731054/FileRecv/resnet34ResizeTwofc/resnet34fourclsWithResize2.pth'))
model4.eval()
model4.cuda()

def get_transforms(phase):
    list_transforms = []

    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    if phase=="resize":
        list_transforms.extend(
            [
                list_transforms.extend([Resize(256, 400, always_apply=True)])  ##记得删除
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
    preds= X_p.numpy().astype('uint8')
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


sample_submission_path = 'E:/data_set/steel/sample_submission.csv'
test_data_folder = "E:/data_set/steel/test_images/"
model.eval()
model.cuda()
threshold=[[0.5,0.5,0.5,0.5]]
test_data=pd.read_csv(sample_submission_path)
cout=np.zeros([4])
sget_transforms = get_transforms('val')
dget_transforms=get_transforms('resize')
for index in range(0, len(test_data),4):

    pic,_=test_data['ImageId_ClassId'][index].split('_')
    imge = cv2.imread(test_data_folder+pic)

    aug = sget_transforms(image=imge)
    img = aug['image'].unsqueeze(0)
    img=img.cuda()

    aug2 = sget_transforms(image=imge)
    img2 = aug2['image'].unsqueeze(0)
    img2 = img2.cuda()

    output=torch.sigmoid(model(img)).data.cpu()
    output2 = torch.sigmoid(model2(img)).data.cpu()
    output3 = torch.sigmoid(model3(img)).data.cpu()
    output4 = torch.sigmoid(model4(img2)).data.cpu()
    output=output*0.5+output2*0.15+output3*0.15+output4*0.2
    pred=(np.copy(output)>threshold)
    # if pred:
    #     submit='1 1'
    # else:
    #     submit=''
    # cv2.imshow(image[1])
    for i in range(4):
        if pred[0][i]:
            test_data['EncodedPixels'][index+i]='1 1'
            cout[i]+=1
        else:
            test_data['EncodedPixels'][index+i]=''

print(cout[0],cout[1],cout[2],cout[3])
test_data.to_csv('E:/pycharm_project/steel/code/submit/fourmodelcls.csv',index=False)


