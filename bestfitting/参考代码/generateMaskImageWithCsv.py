import pandas as pd
import numpy as np

path = "H:/input/severstal-steel-defect-detection/"
in_dir = path + "train_images/"
in_csv_path = path + "train.csv"
out_dir = path + "train_images_mask/"

df = pd.read_csv(in_csv_path)
df.head()

df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

from myutils.mask_functions import pixels2mask, run_length_decode2
import cv2

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
for idx in range(len(df)): 
    image_id, mask = pixels2mask(idx, df)
    mask = mask.astype(np.uint8)
    image_path = in_dir + image_id
    img = cv2.imread(image_path)
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    cv2.imwrite(out_dir + image_id,img)
