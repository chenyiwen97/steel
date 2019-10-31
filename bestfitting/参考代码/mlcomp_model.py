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

from HengResnet import Resnet34_classification
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

import time
start_time = time.time()
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

unet_se_resnext50_32x4d = load('E:/pycharm_project/steel/baseline_model/unet_se_resnext50_32x4d.pth').cuda()
unet_se_resnext50_32x4d.eval()
unet_mobilenet2 = load('E:/pycharm_project/steel/baseline_model/unet_mobilenet2.pth').cuda()
unet_mobilenet2.eval()
unet_resnet34 = load('E:/pycharm_project/steel/baseline_model/unet_resnet34.pth').cuda()
unet_resnet34.eval()

# model_params = {}
# arch = "unet_se_resnext50_cbam_v0a"
# model_params['architecture'] = arch
# unet_se_resnext50_32x4d = init_network(model_params).cuda()
# unet_se_resnext50_32x4d.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/lb89277.pth')['state_dict'])
mymodel=smp.Unet("efficientnet-b0", encoder_weights='imagenet', classes=4, activation=None).cuda()
mymodel.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/b0unetbalanceseg.pth'))
mymodel.eval()
mymodel2= smp.Unet("densenet169", encoder_weights="imagenet", classes=4, activation=None).cuda()
mymodel2.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/densenetseg.pth"))
mymodel2.eval()
mymodel3 = smp.Unet("dpn92", encoder_weights=None, classes=4, activation=None).cuda()
mymodel3.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/dpn92seg.pth"))
mymodel3.eval()

mymodel4 = smp.Unet("efficientnet-b0", encoder_weights=None, classes=5, activation=None).cuda()
mymodel4.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/efficientseg.pth"))
mymodel4.eval()
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


seg_model1 = Model([unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34])
seg_model2=Model([mymodel,mymodel2, mymodel3])

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

model4=Resnet34_classification()
model4.load_state_dict(torch.load('E:/pycharm_project/LB 0.90161/result/resnet34-cls-full-foldb0-0/checkpoint/00007500_model.pth', map_location=lambda storage, loc: storage), strict=True)
model4.cuda()
model4.eval()


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
batch_size = 12
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
min_area = [600, 600, 1000, 2000] # 0.91194
# min_area = [1000,1000,1000,1000]
threshold=0.5 #[[0.5,0.5,0.5,0.5]]
res = [{}]*(4*len(datasets[0]))
idx_res = 0
# Iterate over all TTA loaders
count=np.zeros([4])
total = len(datasets[0]) // batch_size
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    preds = []
    cls_pred=[]
    image_file = []
    for i, batch in enumerate(loaders_batch):
        features = batch['features'].cuda()
        p1 = torch.sigmoid(seg_model1(features))
        p1 = datasets[i].inverse(p1)
        p2 = torch.sigmoid(seg_model2(features))
        p2 = datasets[i].inverse(p2)
        # with torch.no_grad():
        #     p2=torch.softmax(mymodel4(features),dim=1)[:,:-1]
        # p2= datasets[i].inverse(p2)
        # if i==1:
        #     torch.flip(p2,[3])
        p=p1*0.5+p2*0.5
        preds.append(p)
        image_file = batch['image_file']
        with torch.no_grad():
            output = model1(features).data.cpu()
            output2 = model2(features).data.cpu()
            output3 = model3(features).data.cpu()
            output4 = model4(features).data.cpu().reshape(-1,4)
        output=torch.sigmoid(output*0.0959+output2*0.251+output3*0.26+output4*0.393)
        # output = ((output * 1.5)**0.5 + (output2 * 0.75)**0.5 + (output3 * 0.75)**0.5)/3
        # output = (2*(output** 0.5) + (output2) ** 0.5 + (output3) ** 0.5) / 4
        cls_pred.append(output)
        # cls_pred = (np.copy(output) > threshold)
    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()

    cls_pred=torch.stack(cls_pred)
    cls_pred = torch.mean(cls_pred, dim=0)
    cls_pred = (np.copy(cls_pred) > threshold)

    # Batch post processing
    for p_cls,p, file in zip(cls_pred,preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            img_rle = ""
            p_channel = p[i]
            imageid_classid = file + '_' + str(i + 1)
            if p_cls[i]:
                p_channel = (p_channel > thresholds[i]).astype(np.uint8)
                if p_channel.sum() >= min_area[i]:
                    img_rle = mask2rle(p_channel)
                    count[i]+=1
                    # p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            res[idx_res] = {'ImageId_ClassId': imageid_classid, 'EncodedPixels': img_rle}
            idx_res = idx_res + 1
            #else:
                # p_channel = np.zeros(p_channel.shape, dtype=np.uint8)
            # res.append({
            #     'ImageId_ClassId': imageid_classid,
            #     'EncodedPixels': img_rle
            # })

df = pd.DataFrame(res)
columns = ['ImageId_ClassId','EncodedPixels']
df.to_csv('E:/pycharm_project/steel/code/submit/twotta.csv', index=False, columns=columns)
print(count[0],count[1],count[2],count[3])
print("Finished! Spent time about {0} seconds".format(time.time() - start_time))