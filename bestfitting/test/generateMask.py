# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import cv2

path = "H:/input/severstal-steel-defect-detection/"

in_csv_path = path + "train.csv"
out_dir = path + "train_image_mask"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

df = pd.read_csv(in_csv_path)
df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

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

for idx in range(len(df)): 
    image_id, mask = pixels2mask(idx, df)
    mask = mask.astype(np.uint8)*255
    
    for ch in range(4):
        fname = image_id.replace(".", "_" + str(ch) + ".")
        cv2.imwrite(os.path.join(out_dir,fname),mask[:,:,ch])

