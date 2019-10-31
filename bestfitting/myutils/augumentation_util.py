import albumentations as albu
import cv2
from skimage import measure
import random
import numpy as np

# mask 是 二维二值图像
def randomChooseCenter(mask, center_threshold = 70):
    ### ======= 寻找中心点的位置 ============
    labels = measure.label(mask, background=0) # same image_binary as above
    propsa = measure.regionprops(labels)
    center_list = []
    for prop in propsa:
        center_list.append(list(prop.centroid))

    # =========== 对中心点的位置进行进一步的剪裁 =======
    final_center_list = []
    temp_len = len(center_list)

    for idx1 in range(temp_len):
        append_flag = True
        for idx2 in range(idx1 + 1, temp_len):
            if(np.abs(center_list[idx1][1] - center_list[idx2][1]) < center_threshold):
                append_flag = False
                break
        if(append_flag == True):
            final_center_list.append(center_list[idx1])

    #### ===== 获取中心点位置的长度，利用随机数随机选择一个中心点 ========== ######
    final_center_list_len = len(final_center_list)
    randon_center_index = random.randint(0, final_center_list_len - 1)
    chosen_center = final_center_list[randon_center_index]
    return chosen_center

### ============= 模型设计输出 --- 这个idx与random sampler的idx对应 ================ ###
def designValidCropWithIdx(img, mask, class_id, center_threshold = 70):
    ch = (class_id) % 5
    crop_img = np.zeros([256,256,3])
    crop_mask = np.zeros([256,256,4])
    # print("======Sequential channel=========", class_id)
    if(ch==0):
        ## 通道为0, 则选择开始随机裁剪
        assert(np.sum(mask[:,:,:]) == 0)
        aug = albu.RandomCrop(256, 256, always_apply = True)
        augmented = aug(image=img, mask=mask)
        crop_img = augmented['image']
        crop_mask = augmented['mask']
    else:
        assert(np.sum(mask[:,:,ch - 1]) > 0)
        temp_mask = (mask[:,:,ch - 1] > 0).astype(np.uint8)
        chosen_center = randomChooseCenter(temp_mask, center_threshold)
        ###### ====  do crop  ===== ######
        chosen_center = np.array(chosen_center).astype(np.int32)
        ## 只裁剪 == 宽度方向 === 
        if chosen_center[1] <= 128:
            crop_mask = mask[:,0:256,:]
            crop_img = img[:,0:256,:]
        elif chosen_center[1] >= (1600-128):
            crop_mask = mask[:,1600-256:1600,:]
            crop_img = img[:,1600-256:1600,:]
        else:
            crop_mask = mask[:,chosen_center[1]-128:chosen_center[1] + 128,:]
            crop_img = img[:,chosen_center[1]-128:chosen_center[1] + 128,:]
    return crop_img, crop_mask

def designValidCrop(img, mask, center_threshold = 70):
    crop_img = np.zeros([256,256,3])
    crop_mask = np.zeros([256,256,4])
    for ch in range(4):
        temp_mask = (mask[:,:,0] > 0).astype(np.uint8)
        if(np.sum(temp_mask) == 0):
            aug = albu.RandomCrop(256, 256, always_apply = True)
            augmented = aug(image=img[:,:,ch], mask=mask[:,:,ch])
            crop_img[:,:,ch] = augmented['image']
            crop_mask[:,:,ch] = augmented['mask']
        else:
            chosen_center = randomChooseCenter(temp_mask, center_threshold)
            ###### ====  do crop  ===== ######
            chosen_center = np.array(chosen_center).astype(np.int32)
            if chosen_center[1] <= 128:
                crop_mask[:,:,ch] = temp_mask[:,0:256]
                crop_img[:,:,ch] = img
            elif chosen_center[1] >= (1600-128):
                crop_mask[:,:,ch] = temp_mask[:,1600-256:1600]
            else:
                crop_mask[:,:,ch] = temp_mask[:,chosen_center[1]-128:chosen_center[1] + 128]

def dataAugumentation(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = []
    if phase == "train":
        trans.extend(
            [
                # albu.RandomCrop(256, 256, always_apply = True)
                albu.HorizontalFlip(p=0.5), # only horizontal flip as of now
                albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(),
            ]
        )
    trans.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
        ]
    )
    all_trans = albu.Compose(trans)
    return all_trans

def dataAugumentation2(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = []
    if phase == "train":
        trans.extend(
            [
                albu.HorizontalFlip(p=0.5), # only horizontal flip as of now
                albu.VerticalFlip(p=0.5),
                albu.RandomCrop(256, 256, always_apply = True),
                albu.IAAAdditiveGaussianNoise(p=0.2),
                albu.OneOf(
                    [
                        albu.RandomContrast(p=1),
                        albu.HueSaturationValue(p=1),
                    ],
                    p=0.5,
                )
            ]
        )
    trans.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
        ]
    )
    all_trans = albu.Compose(trans)
    return all_trans
def dataAugumentation3(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = [albu.RandomCrop(256, 480, always_apply=True)]
    if phase == "train":
        trans.extend(
            [

                albu.HorizontalFlip(p=0.5), # only horizontal flip as of now
                albu.VerticalFlip(p=0.5),
            ]
        )
    trans.extend(
        [

            albu.Normalize(mean=mean, std=std, p=1),
        ]
    )
    all_trans = albu.Compose(trans)
    return all_trans
def dataAugumentationAfterCrop(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = []
    trans.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
        ]
    )
    all_trans = albu.Compose(trans)
    return all_trans
