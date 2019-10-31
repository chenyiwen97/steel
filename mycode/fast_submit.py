# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
"""这是在kaggle notebook上的，勿在本地跑"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#######################################################################
# kaggle kernel
DATA_DIR  = '../input/severstal-steel-defect-detection'
CSV_FILE  = '../input/20199291552/mergewithjiahao.csv'
SUBMISSION_CSV_FILE = 'submission.csv'

#!ls ../input/steel2019-offline-csv

#########################################################################


df =  pd.read_csv(DATA_DIR + '/sample_submission.csv')
df[ 'EncodedPixels'] = ''

df_predict =  pd.read_csv(CSV_FILE)
for image_id, rle in df_predict.values:
    print('\r %s'%image_id, end='', flush=True)
    df['EncodedPixels'][df['ImageId_ClassId'] == image_id] = rle

df.to_csv(SUBMISSION_CSV_FILE, index=False)