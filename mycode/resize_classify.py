import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
import albumentations as albu
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise, Resize)
from albumentations.torch import ToTensor
from torch.nn.functional import one_hot
def mask2rle(img):
    '''+
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
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

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx].name
        labels = self.df.iloc[idx][:4]
        target=torch.zeros(4)
        for i, label in enumerate(labels.values):
            if not pd.isnull(label):
                target[i]=1
        target=target.float()
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img)
        img = augmented['image']
        return img,target

    def __len__(self):
        return len(self.fnames)

def get_transforms(phase, mean, std):
    list_transforms = []
    list_transforms.extend([Resize(256,400,always_apply=True)])
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(
        data_folder,
        df_path,
        phase,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    df['label'] = df['defects'] > 0
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)     ##??
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds
class Meter():
    def __init__(self):
        self.acc=0
        self.pos_acc=np.zeros(4)
        self.neg_acc=0
        self.num=0
        self.num_neg=0
        self.num_pos=np.zeros(4)

    def update(self,prob,truth,threshold=0.5):
        batch_size = len(truth)
        assert (prob.shape == truth.shape)
        prob=torch.sigmoid(prob)
        prob=np.copy(prob)
        truth=np.copy(truth)
        truth=(truth>0.5).astype(np.float32)
        prob = (prob > threshold).astype(np.float32)
        for i in range(4):
            t=truth[:,i]
            p=prob[:,i]
            su=t+p  #在t+p=1的时候，分类不正确，=2，=0则分类正确
            self.acc=self.acc+sum(su!=1)
            self.num=self.num+len(su)
            self.pos_acc[i]=self.pos_acc[i]+sum(su==2)                          #有缺陷的准预测
            self.neg_acc=self.neg_acc+sum(su==0)                          #没缺陷的准预测
            self.num_pos[i]=self.num_pos[i]+sum(t==1)
            self.num_neg=self.num_neg+sum(t == 0)
    def get_accuracy(self):
        return self.acc/self.num,self.pos_acc/self.num_pos,self.neg_acc/self.num_neg



class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 8, "val": 16}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 50
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # self.optimizer.load_state_dict(torch.load())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.accuracy={phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs


    def iterate(self, epoch, phase):
        meter = Meter()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(outputs,targets)
            acc, pos_acc, neg_acc = meter.get_accuracy()
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        acc,pos_acc,neg_acc=meter.get_accuracy()
        print("Loss: %0.4f |pos_acc1:%0.4f |pos_acc2:%0.4f |pos_acc3:%0.4f |pos_acc4:%0.4f |neg_acc:%0.4f" %
              (epoch_loss,pos_acc[0],pos_acc[1],pos_acc[2],pos_acc[3],neg_acc))
        self.losses[phase].append(epoch_loss)
        self.accuracy[phase].append(neg_acc)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            train_loss=self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.optimizer.state_dict(),
            }
            myplot(self.losses['train'],"train_loss")
            myplot(self.accuracy['train'],"train_acc")

            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
                myplot(self.losses['val'], "validation_loss")
                myplot(self.accuracy['val'], "validation_acc")
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state,"param.pth")
                torch.save(self.net.state_dict(), "./resnet34fourclsWithResize.pth")
                # torch.save(self.net,"./model")
            print()
def myplot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores)), scores, label=f'{name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join('./', name) + ".png")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from efficientnet_pytorch import EfficientNet
    # model = EfficientNet.from_pretrained('efficientnet-b3')
    # fc_features =model._fc.in_features
    # model._fc=nn.Linear(fc_features,4)
    model=models.resnet34(pretrained=True)
    fc_features = model.fc.in_features
    model.fc=nn.Linear(fc_features,4)
    model_trainer = Trainer(model)
    model_trainer.start()

    sample_submission_path = 'E:/pycharm_project/STEEL/sample_submission.csv'
    train_df_path = 'E:/data_set/steel/train.csv'
    data_folder = "E:/data_set/steel/"
    test_data_folder = "E:/data_set/steel/test_images"


