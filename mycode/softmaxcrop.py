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
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
import albumentations as albu
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
"""没有dice loss"""

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
    masks = np.zeros((256, 1600, 5), dtype=np.float32) # float32 is V.Imp
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
    logi=masks[:,:,0]
    for ie in range(1,4):
        logi=np.logical_or(logi,masks[:, :, ie])
        # print(logi.sum())
    masks[:, :, 4]=np.logical_not(logi).astype(np.float32)

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
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
                VerticalFlip(p=0.5),
                # ShiftScaleRotate(),
            ]
        )
    list_transforms.extend(
        [
            albu.RandomCrop(256, 320, always_apply=True),
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
    # df=df[df['defects']!=0]
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    return dataloader
"""应该是要改成取最大值，然后把第5维去掉"""
def predict(X):
    X_p= X >= (torch.max(X, 1)[0].unsqueeze(1))
    X_p=X_p[:,:-1,:,:]
    preds= np.copy(X_p).astype('uint8')
    return preds
def metric_hit(logit, truth, threshold=0.5):
    batch_size,num_class, H,W = logit.shape

    with torch.no_grad():
        x=torch.max(logit,1)[0].unsqueeze(1)
        p = (logit>=x).float()
        p = p.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,-1)
        t = truth
        correct = (p==t)

        index0 = t==0
        index1 = t==1
        index2 = t==2
        index3 = t==3
        index4 = t==4

        num_neg  = index0.sum().item()
        num_pos1 = index1.sum().item()
        num_pos2 = index2.sum().item()
        num_pos3 = index3.sum().item()
        num_pos4 = index4.sum().item()

        neg  = correct[index0].sum().item()/(num_neg +1e-12)
        pos1 = correct[index1].sum().item()/(num_pos1+1e-12)
        pos2 = correct[index2].sum().item()/(num_pos2+1e-12)
        pos3 = correct[index3].sum().item()/(num_pos3+1e-12)
        pos4 = correct[index4].sum().item()/(num_pos4+1e-12)

        num_pos = [num_pos1,num_pos2,num_pos3,num_pos4,]
        tn = neg
        tp = [pos1,pos2,pos3,pos4,]

    return tn,tp, num_neg,num_pos

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        # probability = probability.view(batch_size, -1)

        truth = truth[:,:-1,:,:].view(batch_size, -1)
        x=torch.max(probability,1)[0].unsqueeze(1)
        p = (probability>=x).float()
        p=p[:,:-1,:,:].view(batch_size, -1)
        assert(p.shape == truth.shape)
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)





    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        dice, dice_neg, dice_pos, _, _ = metric(outputs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(outputs)
        iou = compute_iou_batch(preds, targets, classes=[1])               ##??
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    # pred[label == ignore_index] = 0   ##??
    ious = []
    label=label[:-1,:,:]
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    labels = np.copy(labels) # copy is imp
    preds = np.array(outputs) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou





class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 8, "val": 16}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 60
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        # self.criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
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
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        labels=torch.max(targets,1)[1].to(self.device).long()
        labels=labels.unsqueeze(1)
        outputs = self.net(images)
        preds=outputs.permute(0, 2, 3, 1).contiguous().view(-1, 5)
        labels=labels.permute(0, 2, 3, 1).contiguous().view(-1)
        loss = F.cross_entropy(preds, labels,weight=loss_weight)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
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
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state,"./record.pth")
                torch.save(self.net.state_dict(), "./b5cropunetseg.pth")
                # torch.save(self.net,"./model")
            print()

def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
    plt.legend();
    plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # model=torch.load("./model")
    """5类，包括没有的"""
    # model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=5, activation=None)
    model = smp.Unet("efficientnet-b5", encoder_weights="imagenet", classes=5, activation=None)
    model.load_state_dict(torch.load("./b5cropunetseg.pth"))
    sample_submission_path = 'E:/pycharm_project/STEEL/sample_submission.csv'
    train_df_path = 'E:/data_set/steel/train.csv'
    data_folder = "E:/data_set/steel/"
    test_data_folder = "E:/data_set/steel/test_images"
    loss_weight=torch.FloatTensor([1,5,8,2,5]).cuda()
    # loss_weight = torch.FloatTensor([1,40,80,8,40]).cuda()
    model_trainer = Trainer(model)
    sss = torch.load("./record.pth")
    model_trainer.optimizer.load_state_dict(sss["state_dict"])
    # model_trainer.optimizer = optim.Adam(model_trainer.net.parameters(), lr=5e-5)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores # overall dice
    iou_scores = model_trainer.iou_scores



    plot(losses, "BCE loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")