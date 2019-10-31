import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.imageunet import init_network
import cv2
import numpy as np
import pandas as pd

model_params = {}
model_params['architecture'] = "unet_resnet34_cbam_v0a"
model = init_network(model_params)

model.load_state_dict(torch.load("D:/chenyiwen/steel/george/fcn/model_bceabdnormalization.pth"))
model.eval()

path = 'D:/chenyiwen/steel/steel/'

test_data = pd.read_csv(path + 'sample_submission.csv')

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

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

device = torch.device("cuda:0")
model.cuda()
for index in range(0, len(test_data), 4):
    pic, _ = test_data['ImageId_ClassId'][index].split('_')
    img = cv2.imread(path + 'test_images/' + pic)
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).type(torch.FloatTensor).cuda()
    output = model(img)
    pred = predict(output.data.cpu().numpy(), 0.5)
    # cv2.imshow(image[1])
    for i in range(4):
        to_submit=pred[0,i,:,:]
        submit=mask2pixels(to_submit)
        test_data['EncodedPixels'][index+i]=submit

test_data.to_csv("submission.csv",index=False)
