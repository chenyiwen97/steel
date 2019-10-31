############# ====================== ###################
# =========== Main: 
# = two stage 策略： 1. 先进行多分类, 2. 多分类后将其对应的类别利用对应的模型去检测
# #### 一个分类器 - 四个分割器
import pandas as pd
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


data1 = pd.read_csv('C:/Users/PC/Documents/Tencent Files/867731054/FileRecv/clsandfourmodel10131048.csv')
data2 = pd.read_csv('plusclass1.csv')
# data2=pd.read_csv('E:/pycharm_project/steel/code/submit/mergemerge.csv')
count = 0
for index in range(0, len(data1),4):
    if not pd.isnull(data1['EncodedPixels'][index+1]):
        data2['EncodedPixels'][index+1]=data1['EncodedPixels'][index+1]
        count += 1
        print(data1['ImageId_ClassId'][index])
print(count)
data2.to_csv("plusclass12.csv",index=False)





