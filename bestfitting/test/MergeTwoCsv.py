# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def mask2pixels(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pixels2mask(label):
    '''Given a lable, return mask (256, 1600)`'''
    mask = np.zeros(256 * 1600, dtype=np.uint8)
    if not pd.isnull(label):
        label = label.split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        for pos, le in zip(positions, length):
            mask[pos-1:(pos + le-1)] = 1
    mask = mask.reshape(256, 1600, order='F')
    return mask


data1 = pd.read_csv("jiahao089.csv")
data2 = pd.read_csv("mergemerge.csv")

data = data1

assert(len(data1) == len(data2))
for x in range(len(data1)):
    mask1 = pixels2mask(data1["EncodedPixels"][x]) 
    mask2 = pixels2mask(data2["EncodedPixels"][x])
    mask = ((mask1 + mask2) > 0).astye(np.uint8)
    data["EncodedPixels"][x] = mask2pixels(mask)

data.to_csv("zeronineoneUnion.csv", index = False)


