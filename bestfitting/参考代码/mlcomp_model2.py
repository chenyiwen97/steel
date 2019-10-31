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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.jit import load
from torchvision import models
from efficientnet_pytorch import EfficientNet

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap


from networks.imageunet import init_network
from myutils.common_util import *
from layers.scheduler import *
from datasets.datastes import ImageData, MyBalanceClassSampler
from myutils.cal_util import AverageMeter, metric
from myutils.augumentation_util import dataAugumentation

# def post_process(probability, threshold = 0.5, min_size = 200):
#     '''Post processing of each predicted mask, components with lesser number of pixels
#     than `min_size` are ignored'''
#     mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
#     num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
#     predictions = np.zeros((256, 1600), np.float32)
#     for c in range(1, num_component):
#         p = (component == c)
#         if p.sum() > min_size:
#             predictions[p] = 1
#     return predictions

unet_se_resnext50_32x4d = load('E:/pycharm_project/steel/baseline_model/unet_se_resnext50_32x4d.pth').cuda()
unet_mobilenet2 = load('E:/pycharm_project/steel/baseline_model/unet_mobilenet2.pth').cuda()
unet_resnet34 = load('E:/pycharm_project/steel/baseline_model/unet_resnet34.pth').cuda()

# model_params = {}
# arch = "unet_se_resnext50_cbam_v0a"
# model_params['architecture'] = arch
# unet_se_resnext50_32x4d = init_network(model_params).cuda()
# unet_se_resnext50_32x4d.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/lb89277.pth')['state_dict'])
mymodel=smp.Unet("efficientnet-b0", encoder_weights='imagenet', classes=4, activation=None).cuda()
mymodel.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/b0unetbalanceseg.pth'))

mymodel2= smp.Unet("densenet169", encoder_weights="imagenet", classes=4, activation=None).cuda()
mymodel2.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/densenetseg.pth"))
mymodel3 = smp.Unet("dpn92", encoder_weights=None, classes=4, activation=None).cuda()
mymodel3.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/dpn92seg.pth"))
class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)


model = Model([unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34,mymodel,mymodel2])

model1=models.resnet34(pretrained=True)
fc_features = model1.fc.in_features
model1.fc=nn.Linear(fc_features,4)
model1.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet34fourcls.pth'))
model1.cuda()
model1.eval()

model2 = EfficientNet.from_pretrained('efficientnet-b3')
fc_features =model2._fc.in_features
model2._fc=nn.Linear(fc_features,4)
model2.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientnetfourcls.pth'))
model2.cuda()
model2.eval()

model3=models.resnet50(pretrained=True)
fc_features = model3.fc.in_features
model3.fc=nn.Linear(fc_features,4)
model3.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet50fourcls.pth'))
model3.cuda()
model3.eval()

bf_cls_csv = pd.read_csv("E:/KAGGLE_STEEL/OneDrive/kaggle/bestfitting/segtocls/cls.csv")

def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

img_folder = 'E:/data_set/steel/test_images'
batch_size = 2
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]

thresholds = [0.5, 0.7, 0.3, 0.5]
min_area = [800,800,800,800]
threshold=[[0.5,0.5,0.5,0.5]]
res = []
# Iterate over all TTA loaders
total = len(datasets[0]) // batch_size
record_index = 0
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    preds = []
    image_file = []
    for i, batch in enumerate(loaders_batch):
        features = batch['features'].cuda()
        p = torch.sigmoid(model(features))
        # inverse operations for TTA
        p = datasets[i].inverse(p)
        preds.append(p)
        image_file = batch['image_file']
        if i==0:
            output = model1(features).data.cpu()
            output2 = model2(features).data.cpu()
            output3 = model3(features).data.cpu()
            output = torch.sigmoid(output * 0.5 + output2 * 0.25 + output3 * 0.25)
            cls_pred = (np.copy(output) > threshold)
    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()

    # Batch post processing
    for p_cls,p, file in zip(cls_pred,preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            p_channel = p[i]
            imageid_classid = file + '_' + str(i + 1)
            aaa=bf_cls_csv['EncodedPixels'][record_index * 4 + i]
            bbb=p_cls[i]
            if p_cls[i] or (not pd.isna(bf_cls_csv['EncodedPixels'][record_index * 4 + i])):
                p_channel = (p_channel > thresholds[i]).astype(np.uint8)
                if p_channel.sum() < min_area[i]:
                    p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            else:
                p_channel = np.zeros(p_channel.shape, dtype=np.uint8)
            res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })
        record_index += 1

df = pd.DataFrame(res)
# df.to_csv('E:/pycharm_project/steel/code/submit/6seg4cls2.csv', index=False)

columns = ['ImageId_ClassId','EncodedPixels']
df.to_csv('E:/pycharm_project/steel/code/submit/6seg4cls.csv', index=False, columns=columns)