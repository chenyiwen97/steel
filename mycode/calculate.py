import numpy as np
import cv2
import random
import os
# calculate means and std
directory_name="E:/data_set/steel/train_images"
i=0

#计算数据集中图片的mean和std


CNum = 10000  # 挑选多少图片进行计算


imgs = np.zeros([256,1600,1]).astype(np.uint8)
means, stdevs = [], []

for filename in os.listdir(directory_name):

        img_path = os.path.join(directory_name+'/'+filename)
        print(filename, i)
        i = i + 1
        img = cv2.imread(img_path)
        img=np.expand_dims(img[:,:,0],axis=-1)

        imgs = np.concatenate((imgs, img), axis=2)
#         print(i)

imgs = imgs.astype(np.float32) / 255.

pixels = imgs[:, :,  :].ravel()  # 拉成一行
me=np.mean(pixels)
print(me)
st=np.std(pixels)
print(st)


# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
# means.reverse()  # BGR --> RGB
# stdevs.reverse()

