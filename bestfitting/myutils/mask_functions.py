import numpy as np
import pandas as pd
from config.config import *

def run_length_encode(component):
    if component.sum() == 0:
        return '-1'
    component = np.hstack([np.array([0]), component.T.flatten(), np.array([0])])
    start  = np.where(component[1: ] > component[:-1])[0]
    end    = np.where(component[:-1] > component[1: ])[0]
    length = end-start

    rle = []
    for i in range(len(length)):
        if i==0:
            rle.extend([start[0],length[0]])
        else:
            rle.extend([start[i]-end[i-1],length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle

def run_length_decode(rle, height=256, width=1600, fill_value=1):
    component = np.zeros((height,width), np.float32)
    if rle == '-1':
        return component
    component = component.reshape(-1)
    rle  = np.array([int(s) for s in rle.split(' ')])
    rle  = rle.reshape(-1, 2)

    start = 0
    for index,length in rle:
        start = start+index
        end   = start+length
        component[start : end] = fill_value
        start = end

    component = component.reshape(width, height).T
    return component


def run_length_decode2(rle, height=256, width=1600, fill_value=1):
    component = np.zeros((height,width), np.float32)
    try:
        if np.isnan(rle):
            return component
    except TypeError:
        component = component.reshape(-1)
        rle  = np.array([int(s) for s in rle.split(' ')])
        rle  = rle.reshape(-1, 2)

        start = 0
        for index,length in rle:
            start = index - 1
            end   = start+length - 1
            component[start : end] = fill_value
            # start = end

        component = component.reshape(width, height).T
    return component

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

