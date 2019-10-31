import pandas as pd
import numpy as np

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

path = 'H:/input/severstal-steel-defect-detection/'
# path = 'E:/data_set/steel/'

# Data loading code
train_csv = pd.read_csv(path + 'train.csv') # 50272
train_csv['ImageId'], train_csv['ClassId'] = zip(*train_csv['ImageId_ClassId'].str.split('_'))
train_csv['ClassId'] = train_csv['ClassId'].astype(int)
train_csv = pd.pivot(train_csv, index = 'ImageId', columns = 'ClassId', values = 'EncodedPixels')
train_csv['defects'] = train_csv.count(axis=1)

for y in range(1,5):
    sname = 'size_' + str(y)
    train_csv[sname] = len(train_csv) * [np.nan]

for x in range(len(train_csv)):
    if(train_csv['defects'][x] != 0):
        _, mask = pixels2mask(x, train_csv)
        for y in range(4):
            tempSize = np.sum(mask[:,:,y])
            if(tempSize > 0):
                sname = 'size_' + str(y + 1)
                train_csv[sname][x] = tempSize

train_csv.to_csv("allsize.csv", index = False)

for y in range(1,5):
    sname = 'size_' + str(y)
    ss = "min_size_" + str(y)
    print(ss, np.min(train_csv[sname]))
    print()
