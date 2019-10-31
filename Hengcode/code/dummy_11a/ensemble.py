# from common import *
import pandas as pd
import numpy as np

## combile classification and segmentation csv
if 1:
    label_csv ='E:/pycharm_project/steel/code/submit/mergefourclasscls.csv'
    # mask_csv='C:/Users/PC/Downloads/submission.csv'
    mask_csv  ='E:/pycharm_project/steel/code/submit/effiwithbase.csv'
    # mask_csv ='E:/pycharm_project/LB 0.90161/result/resnet18-seg-full-softmax-foldb1-1-4balance/submit/resnet18softmaxtta0.50.csv'

    df_mask = pd.read_csv(mask_csv).fillna('')
    df_label = pd.read_csv(label_csv).fillna('')

    assert(np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))
    print((df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels'] != '').sum() ) #202
    df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels']=''


    csv_file = 'E:/pycharm_project/steel/code/submit/mergemerge.csv'
    df_mask.to_csv(csv_file, columns=['ImageId_ClassId', 'EncodedPixels'], index=False)

