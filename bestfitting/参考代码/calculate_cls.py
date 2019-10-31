# import warnings
# warnings.filterwarnings('ignore')
# import os
# import matplotlib.pyplot as plt
# import segmentation_models_pytorch as smp
# import numpy as np
# import cv2
# import albumentations as A
# from tqdm import tqdm_notebook
# import pandas as pd
# import random
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.jit import load
# from torchvision import models
# from efficientnet_pytorch import EfficientNet
#
# from mlcomp.contrib.transform.albumentations import ChannelTranspose
# from mlcomp.contrib.dataset.classify import ImageDataset
# from mlcomp.contrib.transform.rle import rle2mask, mask2rle
# from mlcomp.contrib.transform.tta import TtaWrap
#
#
# from networks.imageunet import init_network
# from myutils.common_util import *
# from layers.scheduler import *
# from datasets.datastes import ImageData, MyBalanceClassSampler
# from myutils.cal_util import AverageMeter, metric
# from myutils.augumentation_util import dataAugumentation
#
# from HengResnet import Resnet34_classification
# # def post_process(probability, threshold = 0.5, min_size = 200):
# #     '''Post processing of each predicted mask, components with lesser number of pixels
# #     than `min_size` are ignored'''
# #     mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
# #     num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
# #     predictions = np.zeros((256, 1600), np.float32)
# #     for c in range(1, num_component):
# #         p = (component == c)
# #         if p.sum() > min_size:
# #             predictions[p] = 1
# #     return predictions
#
# import time
# start_time = time.time()
# seed = 69
# random.seed(seed)
# os.environ["PYTHONHASHSEED"] = str(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# model1=models.resnet34(pretrained=True)
# fc_features = model1.fc.in_features
# model1.fc=nn.Linear(fc_features,4)
# model1.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet34fourcls.pth'))
# model1.cuda()
# model1.eval()
#
# model2 = EfficientNet.from_pretrained('efficientnet-b3')
# fc_features =model2._fc.in_features
# model2._fc=nn.Linear(fc_features,4)
# model2.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientnetfourcls.pth'))
# model2.cuda()
# model2.eval()
#
# model3=models.resnet50(pretrained=True)
# fc_features = model3.fc.in_features
# model3.fc=nn.Linear(fc_features,4)
# model3.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet50fourcls.pth'))
# model3.cuda()
# model3.eval()
#
# model4=Resnet34_classification()
# model4.load_state_dict(torch.load('E:/pycharm_project/LB 0.90161/result/resnet34-cls-full-foldb0-0/checkpoint/00007500_model.pth', map_location=lambda storage, loc: storage), strict=True)
# model4.cuda()
# model4.eval()
#
#
# def create_transforms(additional):
#     res = list(additional)
#     # add necessary transformations
#     res.extend([
#         A.Normalize(
#             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#         ),
#         ChannelTranspose()
#     ])
#     res = A.Compose(res)
#     return res
#
#
# img_folder = 'E:/data_set/steel/train_images'
# batch_size = 32
# num_workers = 0
#
# # Different transforms for TTA wrapper
# transforms = [
#     [],
#     # [A.HorizontalFlip(p=1)]
# ]
#
# transforms = [create_transforms(t) for t in transforms]
# datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
# loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]
#
# thresholds = [0.5, 0.7, 0.3, 0.5]
# min_area = [600, 600, 1000, 2000] # 0.91194
# # min_area = [1000,1000,1000,1000]
# threshold=0.5 #[[0.5,0.5,0.5,0.5]]
# res = [{}]*(4*len(datasets[0]))
# idx_res = 0
# # Iterate over all TTA loaders
# count=np.zeros([4])
# total = len(datasets[0]) // batch_size
# for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
#     cls_pred=[]
#     image_file = []
#     for i, batch in enumerate(loaders_batch):
#         features = batch['features'].cuda()
#
#         # with torch.no_grad():
#         #     p2=torch.softmax(mymodel4(features),dim=1)[:,:-1]
#         # p2= datasets[i].inverse(p2)
#         # if i==1:
#         #     torch.flip(p2,[3])
#         image_file = batch['image_file']
#         with torch.no_grad():
#             output = model1(features).data.cpu()
#             output2 = model2(features).data.cpu()
#             output3 = model3(features).data.cpu()
#             output4 = model4(features).data.cpu().reshape(-1,4)
#         cls_pred=torch.cat((output,output2,output3,output4),dim=1)
#
#
#     # Batch post processing
#         for p_cls, file in zip(cls_pred, image_file):
#             file = os.path.basename(file)
#             # Image postprocessing
#
#             imageid_classid = file
#             img_rle=np.copy(p_cls)
#             res[idx_res] = {'ImageId_ClassId': imageid_classid, 'EncodedPixels': img_rle}
#             idx_res = idx_res + 1
#
# df = pd.DataFrame(res)
# columns = ['ImageId_ClassId','EncodedPixels']
# df.to_csv('E:/pycharm_project/steel/code/submit/floatcls.csv', index=False, columns=columns)
#
# print("Finished! Spent time about {0} seconds".format(time.time() - start_time))

import pandas as pd
import numpy as np
res=[]
data=pd.read_csv("E:/data_set/steel/train.csv")
for index in range(0,len(data["ImageId_ClassId"]),4):
    label=np.zeros([4])
    imageid_classid = data["ImageId_ClassId"][index]
    for i in range(4):

        if not pd.isna(data["EncodedPixels"][index+i]):
            label[i]=1
    res.append({'ImageId_ClassId': imageid_classid,'true label':label})
res = pd.DataFrame(res)
res.to_csv('E:/pycharm_project/steel/code/submit/truelabel.csv', index=False)
