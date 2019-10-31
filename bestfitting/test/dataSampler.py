# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from torch.utils.data.sampler import Sampler

path = 'H:/input/severstal-steel-defect-detection/'
train_csv = pd.read_csv(path + 'train.csv')
train_csv['ImageId'], train_csv['ClassId'] = zip(*train_csv['ImageId_ClassId'].str.split('_'))
train_csv['ClassId'] = train_csv['ClassId'].astype(int)
train_csv = pd.pivot(train_csv, index = 'ImageId', columns = 'ClassId', values = 'EncodedPixels')
train_csv['defects'] = train_csv.count(axis=1)

train_csv['first'] = 1 - pd.isnull(train_csv[1]).astype(np.int32)
train_csv['second'] = 1 - pd.isnull(train_csv[2]).astype(np.int32)
train_csv['third'] = 1 - pd.isnull(train_csv[3]).astype(np.int32)
train_csv['forth'] = 1 - pd.isnull(train_csv[4]).astype(np.int32)

neg_len = np.sum(train_csv['defects'] == 0)
neg_index = np.array(np.where(train_csv['defects'] == 0))
pos1_index = np.array(np.where(train_csv['first'] == 1))
pos2_index = np.array(np.where(train_csv['second'] == 1))
pos3_index = np.array(np.where(train_csv['third'] == 1))
pos4_index = np.array(np.where(train_csv['forth'] == 1))


### 另一种获取方式
train_csv['sample'] = 1
pos2_index = np.intersect1d(np.array(np.where(train_csv['second'] == 1)), np.array(np.where(train_csv['sample'] == 1)))
train_csv['sample'][pos2_index] = 0
pos4_index = np.intersect1d(np.array(np.where(train_csv['forth'] == 1)), np.array(np.where(train_csv['sample'] == 1)))
train_csv['sample'][pos4_index] = 0
pos1_index = np.intersect1d(np.array(np.where(train_csv['first'] == 1)), np.array(np.where(train_csv['sample'] == 1)))
train_csv['sample'][pos1_index] = 0
pos3_index = np.intersect1d(np.array(np.where(train_csv['third'] == 1)), np.array(np.where(train_csv['sample'] == 1)))
train_csv['sample'][pos3_index] = 0


neg_len = np.sum(train_csv['defects'] == 0)
neg_index = np.array(np.where(train_csv['defects'] == 0))


neg  = np.random.choice(neg_index.reshape(-1), neg_len, replace=False)
pos1 = np.random.choice(pos1_index.reshape(-1), neg_len, replace=True) # 差1是怎么回事
pos2 = np.random.choice(pos2_index.reshape(-1), neg_len, replace=True)
pos3 = np.random.choice(pos3_index.reshape(-1), neg_len, replace=True)
pos4 = np.random.choice(pos4_index.reshape(-1), neg_len, replace=True)

l = np.stack([neg, pos1, pos2, pos3, pos4]).T
l = l.reshape(-1)

'''

'''

'''
neg_index = np.where(record_label==0)[0]
neg  = np.random.choice(record_label==0, [L,6], replace=True)
pos1 = np.random.choice(record_label==1, L, replace=True)
pos2 = np.random.choice(record_label==2, L, replace=True)
pos3 = np.random.choice(record_label==3, L, replace=True)
pos4 = np.random.choice(record_label==4, L, replace=True)
l = np.stack([neg.reshape, pos1, pos2, pos3, pos3, pos4]).T
l = l.reshape(-1)
l = l[:self.length]
'''

class FourBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4*num_neg

    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length



import numpy as np

a = np.random.uniform(-1,1,(256,256,3)) / 2
    