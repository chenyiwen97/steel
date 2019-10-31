import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from myutils.mask_functions import *
from myutils.augment_util import *
from myutils.common_util import *
from myutils.augumentation_util import designValidCropWithIdx

import numpy as np
import cv2

class ImageData(Dataset):
    def __init__(self, df, path, augmentation, mean, std, subset="train"):
        super().__init__()
        self.df = df
        if augmentation != None:
            self.augmentation = augmentation(subset, mean, std)
        else:
            self.augmentation = None
        self.subset = subset
        
        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "valid":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
    
    @staticmethod
    def numpy_to_torch(s):
        return torch.from_numpy(s.astype(np.float32))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):     
        imgId, mask = pixels2mask(idx, self.df)
        imgPath = self.data_path + imgId
        # print(imgPath)
        # img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(imgPath)    
        if self.augmentation != None:             
            augmented = self.augmentation(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        img = np.transpose(img, [2, 0, 1])
        # img = np.expand_dims(img, axis=0)
        mask = np.transpose(mask, [2, 0, 1])
        # mask = np.expand_dims(mask, axis=0)  
        # mask = mask[0].permute(2,0,1) # 1x4x256x1600
        x = self.numpy_to_torch(img)
        y = self.numpy_to_torch(mask)
        return x, y

class ImageDataWithValidCrop(Dataset):
    # 调用这个类的时候 augmentation 中不能有 crop操作
    def __init__(self, df, path, augmentation, mean, std, subset="train"):
        super().__init__()
        self.df = df
        if augmentation != None:
            self.augmentation = augmentation(subset, mean, std)
        else:
            self.augmentation = None
        self.subset = subset
        
        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "valid":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'

        self.counter = 0
    
    @staticmethod
    def numpy_to_torch(s):
        return torch.from_numpy(s.astype(np.float32))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):     
        imgId, mask = pixels2mask(idx, self.df)
        imgPath = self.data_path + imgId
        class_id = self.df.iloc[idx]["defects"]
        # print(imgPath)
        # img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(imgPath)

        if self.subset == "train":
            img, mask = designValidCropWithIdx(img, mask, self.counter)
            self.counter += 1
        elif self.subset == "valid":
            img, mask = designValidCropWithIdx(img, mask, self.counter)
            self.counter += 1
        elif self.subset == "test":
            pass
        if self.augmentation != None:             
            augmented = self.augmentation(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        img = np.transpose(img, [2, 0, 1])
        # img = np.expand_dims(img, axis=0)
        mask = np.transpose(mask, [2, 0, 1])
        # mask = np.expand_dims(mask, axis=0)  
        # mask = mask[0].permute(2,0,1) # 1x4x256x1600
        x = self.numpy_to_torch(img)
        y = self.numpy_to_torch(mask)
        return x, y

class ImageDataBalanceLoader(Dataset):
    def __init__(self, df, path, augmentation, mean, std, subset="train"):
        super().__init__()
        self.df = df
        if augmentation != None:
            self.augmentation = augmentation(subset, mean, std)
        else:
            self.augmentation = None
        self.subset = subset
        
        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "valid":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
    
    @staticmethod
    def numpy_to_torch(s):
        return torch.from_numpy(s.astype(np.float32))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):     
        imgId, mask = pixels2mask(idx, self.df)
        imgPath = self.data_path + imgId
        # print(imgPath)
        # img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(imgPath)    
        if self.augmentation != None:             
            augmented = self.augmentation(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        img = np.transpose(img, [2, 0, 1])
        # img = np.expand_dims(img, axis=0)
        mask = np.transpose(mask, [2, 0, 1])
        # mask = np.expand_dims(mask, axis=0)  
        # mask = mask[0].permute(2,0,1) # 1x4x256x1600
        x = self.numpy_to_torch(img)
        y = self.numpy_to_torch(mask)
        return x, y

class MyBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset.df
        self.dataset['first'] = 1 - pd.isnull(self.dataset[1]).astype(np.int32)
        self.dataset['second'] = 1 - pd.isnull(self.dataset[2]).astype(np.int32)
        self.dataset['third'] = 1 - pd.isnull(self.dataset[3]).astype(np.int32)
        self.dataset['forth'] = 1 - pd.isnull(self.dataset[4]).astype(np.int32)
        neg_len = np.sum(self.dataset['defects'] == 0)
        self.length = 5*neg_len
        neg_index = np.array(np.where(self.dataset['defects'] == 0))
        pos1_index = np.array(np.where(self.dataset['first'] == 1))
        pos2_index = np.array(np.where(self.dataset['second'] == 1))
        pos3_index = np.array(np.where(self.dataset['third'] == 1))
        pos4_index = np.array(np.where(self.dataset['forth'] == 1))
        neg  = np.random.choice(neg_index.reshape(-1), neg_len, replace=False)
        pos1 = np.random.choice(pos1_index.reshape(-1), neg_len, replace=True) # 差1是怎么回事
        pos2 = np.random.choice(pos2_index.reshape(-1), neg_len, replace=True)
        pos3 = np.random.choice(pos3_index.reshape(-1), neg_len, replace=True)
        pos4 = np.random.choice(pos4_index.reshape(-1), neg_len, replace=True)
        l = np.stack([neg, pos1, pos2, pos3, pos4]).T
        self.l = l.reshape(-1)

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return self.length

class MyBalanceClassSampler2(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset.df
        self.dataset['first'] = 1 - pd.isnull(self.dataset[1]).astype(np.int32)
        self.dataset['second'] = 1 - pd.isnull(self.dataset[2]).astype(np.int32)
        self.dataset['third'] = 1 - pd.isnull(self.dataset[3]).astype(np.int32)
        self.dataset['forth'] = 1 - pd.isnull(self.dataset[4]).astype(np.int32)
        neg_len = np.sum(self.dataset['forth'] == 1)
        self.length = 5*neg_len
        neg_index = np.array(np.where(self.dataset['defects'] == 0))
        pos1_index = np.array(np.where(self.dataset['first'] == 1))
        pos2_index = np.array(np.where(self.dataset['second'] == 1))
        pos3_index = np.array(np.where(self.dataset['third'] == 1))
        pos4_index = np.array(np.where(self.dataset['forth'] == 1))
        neg  = np.random.choice(neg_index.reshape(-1), neg_len, replace=False)
        pos1 = np.random.choice(pos1_index.reshape(-1), neg_len, replace=False) # 差1是怎么回事
        pos2 = np.random.choice(pos2_index.reshape(-1), neg_len, replace=True)
        pos3 = np.random.choice(pos3_index.reshape(-1), neg_len, replace=False)
        pos4 = np.random.choice(pos4_index.reshape(-1), neg_len, replace=False)
        l = np.stack([neg, pos1, pos2, pos3, pos4]).T
        self.l = l.reshape(-1)

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return self.length

'''
class SteelDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.uid = list(np.concatenate([np.load(DATA_DIR + '/split/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s'%f) for f in csv])
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in self.uid for c in [1,2,3,4] ])
        self.df = df

    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()

        length = len(self)
        num = len(self)*4
        pos = (self.df['Label']==1).sum()
        neg = num-pos



    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])

        infor = Struct(
            index    = index,
            folder   = folder,
            image_id = image_id,
        )

        if self.augment is None:
            return image, mask, infor
        else:
            return self.augment(image, mask, infor)

'''