import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.imageunet import init_network
import cv2
import numpy as np
import pandas as pd
from torchvision import models
import albumentations as albu
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp
# model_params = {}
# model_params['architecture'] = "unet_se_resnext50_cbam_v0a"
# seg_model = init_network(model_params)
from torch.jit import load
from albumentations.pytorch.transforms import ToTensor
unet_se_resnext50_32x4d = load('E:/pycharm_project/steel/baseline_model/unet_se_resnext50_32x4d.pth').cuda()
unet_se_resnext50_32x4d.eval()
unet_mobilenet2 = load('E:/pycharm_project/steel/baseline_model/unet_mobilenet2.pth').cuda()
unet_mobilenet2.eval()
unet_resnet34 = load('E:/pycharm_project/steel/baseline_model/unet_resnet34.pth').cuda()
unet_resnet34.eval()

# model_params = {}
# arch = "unet_se_resnext50_cbam_v0a"
# model_params['architecture'] = arch
# unet_se_resnext50_32x4d = init_network(model_params).cuda()
# unet_se_resnext50_32x4d.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/lb89277.pth')['state_dict'])
mymodel=smp.Unet("efficientnet-b0", encoder_weights='imagenet', classes=4, activation=None).cuda()
mymodel.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/b0unetbalanceseg.pth'))
mymodel.eval()
mymodel2= smp.Unet("densenet169", encoder_weights="imagenet", classes=4, activation=None).cuda()
mymodel2.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/densenetseg.pth"))
mymodel2.eval()
mymodel3 = smp.Unet("dpn92", encoder_weights=None, classes=4, activation=None).cuda()
mymodel3.load_state_dict(torch.load("E:/pycharm_project/steel/code/save_model/dpn92seg.pth"))
mymodel3.eval()
class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

model = Model([unet_se_resnext50_32x4d,unet_mobilenet2, unet_resnet34,mymodel,mymodel2,mymodel3])

model1=models.resnet34(pretrained=True)
fc_features = model1.fc.in_features
model1.fc=nn.Linear(fc_features,4)
model1.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet34fourcls.pth'))
model1.cuda()
model1.eval()

model2 = EfficientNet.from_pretrained('efficientnet-b3')
fc_features =model2._fc.in_features
model2._fc=nn.Linear(fc_features,4)
model2.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientnetfourcls.pth'))
model2.cuda()
model2.eval()

model3=models.resnet50(pretrained=True)
fc_features = model3.fc.in_features
model3.fc=nn.Linear(fc_features,4)
model3.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet50fourcls.pth'))
model3.cuda()
model3.eval()

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc_features = model.fc.in_features
        self.fc = nn.Sequential(nn.Linear(self.fc_features, 32),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(32, 4)
                                    )
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# seg_model = smp.Unet("efficientnet-b1", encoder_weights='imagenet', classes=4, activation=None)
# seg_model.load_state_dict(torch.load("E:/pycharm_project/steel/code/3512b1.pth"))
# seg_model.eval()

# cls_model=models.resnet34(pretrained=True)
# fc_features = cls_model.fc.in_features
# cls_model.fc=nn.Linear(fc_features,4)
# cls_model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/resnet34fourcls.pth'))

# cls_model = EfficientNet.from_pretrained('efficientnet-b3')
# fc_features =cls_model._fc.in_features
# cls_model._fc=nn.Linear(fc_features,1)
# cls_model.load_state_dict(torch.load('E:/pycharm_project/steel/code/save_model/efficientnetcls.pth'))
# cls_model.eval()
path = 'E:/data_set/steel/'

test_data = pd.read_csv(path + 'sample_submission.csv')


def dataAugumentation(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = []
    if phase == "train":
        trans.extend(
            [
                # albu.RandomCrop(256, 256, always_apply = True)
                albu.HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    if phase == "resize":
        trans.extend(
            [
                albu.Resize(256,400),
                # albu.HorizontalFlip(p=1), # only horizontal flip as of now
            ]
        )
    trans.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    all_trans = albu.Compose(trans)
    return all_trans

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
    for i in range(4):
        X_p[0,i,:,:]=X_p[0,i,:,:]>threshold[i]
    preds = X_p.astype('uint8')
    return preds

def post_process(probability, threshold = 0.5, min_size = 200):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return predictions

min_area = [600, 600, 1000, 2000]
device = torch.device("cuda:0")
# seg_model.cuda()
# cls_model.cuda()
threshold=[[0.5,0.5,0.5,0.5]]
seg_threshold=[0.5,0.7,0.3,0.5]
count=np.zeros([4])
augmentation = dataAugumentation('test')
for index in range(0, len(test_data), 4):
    pic, _ = test_data['ImageId_ClassId'][index].split('_')
    img = cv2.imread(path + 'test_images/' + pic)
    augmented = augmentation(image=img)
    img = augmented['image'].unsqueeze(0).cuda()
    output = torch.sigmoid(model(img)).data.cpu()                #要不要加sigmoid
    img2=torch.flip(img,[3])
    output2 =torch.sigmoid(model(img2)).data.cpu()
    output2=torch.flip(output2,[3])
    output=output*0.5+output2*0.5
    pred = predict(output.numpy(),seg_threshold )
    with torch.no_grad():
        cls_output1 = model1(img).data.cpu()
        cls_output2 = model2(img).data.cpu()
        cls_output3 = model3(img).data.cpu()
        output = torch.sigmoid(cls_output1 * 0.5 + cls_output2 * 0.25 + cls_output3 * 0.25)
        cls_pred = (np.copy(output) > threshold)

    # cls=torch.sigmoid(cls_model(img))
    # cls=cls.data.cpu().numpy()
    # cv2.imshow(image[1])
    # threshold_pixel = [0.5,0.5,0.6,0.5]

    for i in range(4):
        if cls_pred[0,i]:
            c=pred[0,i]
            if pred[0,i].sum() < min_area[i]:
                # pred[i] = np.zeros(pred[i].shape, dtype=pred[i].dtype)
                submit = ''
            else:
                submit = mask2pixels(pred[0,i])

            if submit!='':
                count[i]+=1
            test_data['EncodedPixels'][index + i] = submit
        else:
            test_data['EncodedPixels'][index + i] =''
print(count[0],count[1],count[2],count[3])
test_data.to_csv("E:/pycharm_project/steel/code/submit/withmobile.csv", index=False)
