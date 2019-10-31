# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
from skimage import measure
import random

## 主要目的是想找出mask中所有缺陷位置的中心，然后随机选取一个中心的位置开始剪裁

img = cv2.imread("./test/0b7a4c9b9_2.jpg", cv2.IMREAD_GRAYSCALE)
img = (img > 0).astype(np.uint8)

startTime = time.time()
# 设置一个 center_threshold
center_threshold = 70

### ======= 寻找中心点的位置 ============
labels = measure.label(img, background=0) # same image_binary as above
propsa = measure.regionprops(labels)
center_list = []
for prop in propsa:
    print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
    print(prop.centroid)
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
print(time.time() - startTime)

#### ===== 获取中心点位置的长度，利用随机数随机选择一个中心点 ========== ######
final_center_list_len = len(final_center_list)
randon_center_index = random.randint(0,final_center_list_len - 1)

chosen_center = final_center_list[randon_center_index]

mask = img

###### ====  crop mask  ===== ######
crop_mask = np.zeros([256,256])
chosen_center = np.array(chosen_center).astype(np.int32)
if chosen_center[1] <= 128:
    crop_mask = mask[:,0:256]
elif chosen_center[1] >= (1600-128):
    crop_mask = mask[:,1600-256:1600]
else:
    crop_mask = mask[:,chosen_center[1]-128:chosen_center[1] + 128]



