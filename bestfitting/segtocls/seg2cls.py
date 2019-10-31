# -*- coding: utf-8 -*-

import pandas as pd

seg_csv = pd.read_csv("lb89277.csv")

for x in range(len(seg_csv)):
    if pd.isna(seg_csv["EncodedPixels"][x]):
        seg_csv["EncodedPixels"][x] = ''
    else:
        seg_csv["EncodedPixels"][x] = '1 1'

seg_csv.to_csv("cls.csv", index = False)



import pandas as pd
cls_csv = pd.read_csv("cls.csv")
seg_csv = pd.read_csv("lb91215.csv")
for x in range(len(seg_csv)):
    if cls_csv["EncodedPixels"][x] == '1 1':
        cls_csv["EncodedPixels"][x] = seg_csv["EncodedPixels"][x]
cls_csv.to_csv("seg913.csv", index = False)
