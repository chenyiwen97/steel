import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



cls_path = 'E:/data_set/steel/sample_submission.csv'
# seg_path='E:/pycharm_project/LB 0.90161/result/resnet18-seg-full-softmax-foldb1-1-4balance/submit/resnet18softmaxtta0.50.csv'
seg_path='E:/pycharm_project/steel/code/submit/jiahao089.csv'
save_csv_path='E:/pycharm_project/steel/code/submit/mergewithjiahao.csv'
test_data=pd.read_csv(seg_path)
ver_data=pd.read_csv(cls_path)

for index in range(0, len(ver_data),4):
    assert ver_data['ImageId_ClassId'][index]==test_data['ImageId_ClassId'][index]
    print(test_data['ImageId_ClassId'][index])
    c=ver_data['EncodedPixels'][index]
    if pd.isnull(ver_data['EncodedPixels'][index]):
        for i in range(4):
            test_data['EncodedPixels'][index+i]=''


test_data.to_csv(save_csv_path,index=False)
