import sys
sys.path.append("..")
import numpy as np
import albumentations as albu
import cv2

from myutils.mask_functions import pixels2mask, run_length_decode2
import matplotlib.pyplot as plt
import os
import pandas as pd

def imgAug():    
    list_trans = []    
    list_trans.extend([
        # step 1
        #albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply= True),
        albu.Normalize(mean=(0.3435, 0.3435, 0.3435), std=(0.1934, 0.1934, 0.1934), always_apply= True),
        # # # albu.RandomBrightness(always_apply=True),
        ])
    return albu.Compose(list_trans)

# 需要保存的图片的数量
def saveFig(img, save_file_name, num):
    plt.figure()
    plt.xticks([])  #去掉x轴
    plt.yticks([])  #去掉y轴                                            
    plt.axis('off')  #去掉坐标轴
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    for x in range(1,num + 1):
        aug = imgAug()
        img_aug = aug(image = img)['image']
        plt.subplot(2,5,x)
        plt.xticks([])  #去掉x轴
        plt.yticks([])  #去掉y轴
        plt.axis('off')  #去掉坐标轴
        plt.imshow(img_aug)
    plt.savefig(save_file_name)      

if __name__ == "__main__":

    # file_dir = "./test/selectPic"
    path = "H:/input/severstal-steel-defect-detection"
    file_dir = os.path.join(path, "train_images")
    file_save_dir = os.path.join("H:/input/severstal-steel-defect-detection", "normalizeExample2")
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)

    in_csv_path = os.path.join(path, "train.csv")
    df = pd.read_csv(in_csv_path)

    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

    saveImg = np.zeros([256*4,1600,3])

    picLen = 100 # len(df)

    # 缺陷标记颜色设置
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
    for _, _, files in os.walk(file_dir):    
        # print(files) #当前路径下所有非目录子文件
        for idx in range(picLen):
            image_id, mask = pixels2mask(idx, df)
            file_name = os.path.join(file_dir, image_id)
            img = cv2.imread(file_name)
            file_save_name = os.path.join(file_save_dir, image_id)
            # file_name = file_save_name.replace(".jpg", "withNor.jpg")
            # saveFig(img, file_save_name, 1)
            augfuc = imgAug()
            aug = augfuc(image = img, mask = mask) 
            img_aug, mask = aug['image'], aug['mask']
            mask = mask.astype(np.uint8)
            img_aug = img_aug*255 # 返回-255 ~ 255的范围
            saveImg[:256,:,:] = img
            saveImg[256*2:256*3,:,:] = img_aug
            # 生成 缺陷标记
            for ch in range(4):
                contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for i in range(0, len(contours)):
                    cv2.polylines(img, contours[i], True, palet[ch], 2)
                    cv2.polylines(img_aug, contours[i], True, palet[ch], 2)
            saveImg[256:256*2,:,:] = img
            saveImg[256*3:256*4,:,:] = img_aug          
            cv2.imwrite(file_save_name, saveImg)


