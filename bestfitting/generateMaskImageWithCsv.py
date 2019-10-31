import pandas as pd
import numpy as np
import os
from skimage import measure

path = "H:/input/severstal-steel-defect-detection/"
in_dir = path + "test_images/"
in_csv_path = path + "plusclass1.csv"
out_dir = path + "test_images_mask91_close"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

df = pd.read_csv(in_csv_path)
df.head()
close_df = pd.read_csv(in_csv_path)

df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

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

from myutils.mask_functions import pixels2mask
import cv2

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
for idx in range(len(df)): 
    image_id, mask = pixels2mask(idx, df)
    mask = mask.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    
    image_path = in_dir + image_id
    img = cv2.imread(image_path)
    for ch in range(4):
        closing = cv2.morphologyEx(mask[:, :, ch], cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
        close_df['EncodedPixels'][idx + ch] = mask2pixels(closing)
    cv2.imwrite(os.path.join(out_dir,image_id),img)

close_df.to_csv("closewithyiwenpluscls1.csv", index = False)
    
