import pandas as pd
import numpy as np
## combile classification and segmentation csv

label_csv ='./ensemble/data/resnet34-cls-tta-0.50.csv'
mask_csv  ='./ensemble/data/segementation.csv'
df_mask = pd.read_csv(mask_csv).fillna('')
df_label = pd.read_csv(label_csv).fillna('')

assert(np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))
print((df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels'] != '').sum() ) #202
df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels']=''

csv_file = './ensemble/data/mergesubmission2.csv'
df_mask.to_csv(csv_file, columns=['ImageId_ClassId', 'EncodedPixels'], index=False)