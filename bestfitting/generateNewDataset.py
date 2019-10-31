import pandas as pd
import numpy as np
import os
import cv2
from skimage import measure

# mask 是 二维二值图像
def allChooseCenter(mask, center_threshold = 70):
    ### ======= 寻找中心点的位置 ============
    labels = measure.label(mask, background=0) # same image_binary as above
    propsa = measure.regionprops(labels)
    center_list = []
    for prop in propsa:
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
    return final_center_list

def validCropForSingleCls(img, mask, chosen_center):
    w_size = 512 
    # 这个主要是针对 有1 2 3 4类缺陷的图片进行裁剪
    crop_img = np.zeros([256,w_size,3])
    crop_mask = np.zeros([256,w_size])

    ###### ====  do crop  ===== ######
    chosen_center = np.array(chosen_center).astype(np.int32)
    ## 只裁剪 == 宽度方向 === 
    if chosen_center[1] <= (w_size // 2):
        crop_mask = mask[:,0:w_size]
        crop_img = img[:,0:w_size,:]
    elif chosen_center[1] >= (1600-(w_size // 2)):
        crop_mask = mask[:,1600-w_size:1600]
        crop_img = img[:,1600-w_size:1600,:]
    else:
        crop_mask = mask[:,chosen_center[1]-(w_size // 2):chosen_center[1] + (w_size // 2)]
        crop_img = img[:,chosen_center[1]-(w_size // 2):chosen_center[1] + (w_size // 2),:]
    return crop_img, crop_mask

def pixels2maskWithIndex(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600) from the dataframe `df`'''

    label = df["EncodedPixels"][row_id]

    mask = np.zeros(256 * 1600, dtype=np.uint8)
    if not pd.isnull(label):
        label = label.split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        for pos, le in zip(positions, length):
            mask[pos-1:(pos + le-1)] = 1
    mask = mask.reshape(256, 1600, order='F')
    return mask

def main(cls_label):

    # Data loading code
    train_csv = pd.read_csv(os.path.join(path, 'train.csv')) # 50272
    
    train_csv['ImageId'], train_csv['ClassId'] = zip(*train_csv['ImageId_ClassId'].str.split('_'))
    train_csv['ClassId'] = train_csv['ClassId'].astype(int)
    train_csv = train_csv[(pd.isnull(train_csv['EncodedPixels']).astype(np.int32) == 0)]
    train_csv = train_csv[(train_csv['ClassId']).astype(np.int32) == cls_label]
    # print(train_csv.head())

    #import ipdb; ipdb.set_trace()
    cropImgPath = os.path.join(newDateSet_path, "crop_images")
    cropMaskPath = os.path.join(newDateSet_path, "crop_masks")
    cropImgWithMaskPath = os.path.join(newDateSet_path, "cropImgAndMask")
    if not os.path.exists(cropImgPath):
        os.makedirs(cropImgPath)
    if not os.path.exists(cropMaskPath):
        os.makedirs(cropMaskPath)
    if not os.path.exists(cropImgWithMaskPath):
        os.makedirs(cropImgWithMaskPath)


    save_index = 0
    for index, row in train_csv.iterrows():
        image_id, encoded_pixels = row['ImageId'], row['EncodedPixels']
        img = cv2.imread(os.path.join(path + "/train_images", image_id))

        # Todo 根据img得到mask
        mask = pixels2maskWithIndex(index, train_csv)

        temp_mask = (mask[:,:] > 0).astype(np.uint8)
        if int(cls_label) == 2:
            all_chosen_center = allChooseCenter(temp_mask, 10)
        else:
            all_chosen_center = allChooseCenter(temp_mask)
        print(len(all_chosen_center))
        # 遍历list
        for chosen_center in all_chosen_center:

            crop_img, crop_mask = validCropForSingleCls(img, mask, chosen_center)
            # 保存裁剪的图片
            cv2.imwrite(cropImgPath + "/" + str(save_index) + ".png", crop_img)
            cv2.imwrite(cropMaskPath + "/" + str(save_index) + ".png", crop_mask*255)
            palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
        
            contours, _ = cv2.findContours(crop_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for i in range(0, len(contours)):
                cv2.polylines(crop_img, contours[i], True, palet[cls_label - 1], 2)
        
            cv2.imwrite(cropImgWithMaskPath + "/" + str(save_index) + ".png", crop_img)
            save_index = save_index + 1

    #     ImageId_ClassId, has_defect = row['ImageId_ClassId'], row['EncodedPixels']
        # print(ImageId_ClassId, has_defect)
        # if not pd.isna(has_defect):
        #     image_id, class_id = ImageId_ClassId.split("_")
        #     img = cv2.imread(os.path.join(path + "/test_images", image_id))

            
        #     augmentation = dataAugumentation('test')
        #     augmented = augmentation(image=img)
        #     img = augmented['image']
        #     img = np.transpose(img, [2, 0, 1])
        #     img = np.expand_dims(img, axis=0)
        #     img = torch.from_numpy(img).type(torch.FloatTensor).cuda()
        #     if class_id == "1":
        #         output = model_cls1(img)
        #     elif class_id == "2":
        #         output = model_cls2(img)
        #     elif class_id == "3":
        #         output = model_cls3(img)            
        #     elif class_id == "4":
        #         output = model_cls4(img)
        #     output = F.sigmoid(output)
        #     pred = post_process(predict(output.data.cpu().numpy(), 0.5))
        #     submit = mask2pixels(pred)
        #     test_data['EncodedPixels'][index] = submit
    # import ipdb; ipdb.set_trace()


path = "D:/chenyiwen/steel/steel"
newDateSet_path = "E:/george/newSteelDataset/class"


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    for cls_label in [2]:
        newDateSet_path = "E:/george/newSteelDataset/class" + str(cls_label)
        if not os.path.exists(newDateSet_path):
            os.makedirs(newDateSet_path)
        main(cls_label)
        print("\n Class " + str(cls_label) + " generated successfully!")
    print("\nfinish all!")
    