import torch
import torch.nn as nn
from torchvision.transforms import transforms

from model.W_Ore_Net import Unet13sum
import cv2 as cv

import os
import numpy as np
import time
from labelme import utils
import os.path as osp
from PIL import Image
import PIL
import imgviz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_train_best = './pth/1iou.pth'

model_pre = Unet13sum(3, 6)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model_pre = nn.DataParallel(model_pre)
model_pre.to(device)
model_pre.load_state_dict(torch.load(path_train_best))

val_predict_path = './test/img256'
val_predict = './test/predict'
label_path2 = './test/label256'

img_path = []  # 预测图片的地址
label_path = []
n = len(os.listdir(val_predict_path))
for j in range(n):
    img1 = os.path.join(val_predict_path, "%d.png" % j)
    predict = os.path.join(val_predict, '%d.png' % j)
    label1 = os.path.join(label_path2, '%d.png' % j)
    img_path.append(img1)
    label_path.append(label1)

total_he = 0.0
iou_he = 0.0
presion_he = 0.0
recall_he = 0.0
under_seg_he = 0.0
over_seg_he = 0.0
f1score_he = 0.0

he_iou1 = 0.0
he_precision1 = 0.0
he_recall1 = 0.0

he_iou2 = 0.0
he_precision2 = 0.0
he_recall2 = 0.0

he_iou3 = 0.0
he_precision3 = 0.0
he_recall3 = 0.0

he_iou4 = 0.0
he_precision4 = 0.0
he_recall4 = 0.0

he_iou5 = 0.0
he_precision5 = 0.0
he_recall5 = 0.0

time1 = time.time()
model_pre.eval()
with torch.no_grad():
    for i in range(n):
        img = Image.open(os.path.join(val_predict_path, '%d.png' % i))
        filename = osp.join(val_predict, '%d.png' % i)
        label = Image.open(os.path.join(label_path2, '%d.png' % i))
        label = np.array(label, dtype='int64')

        img_predict = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(img)

        img_predict = torch.unsqueeze(img_predict, 0)
        _, y = model_pre(img_predict.to(device)) # 单loss
        y = torch.argmax(y, 1)

        y = y.cpu().detach()
        img_y = torch.squeeze(y).numpy()
        lbl_pil = PIL.Image.fromarray(img_y.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)

        he = np.zeros((256, 256))
        he[img_y == label] = 1
        he = he.sum()
        mean_he = he/(256*256)
        total_he = total_he + mean_he

        tp = np.sum(img_y[label == 1] == 1)
        fp = np.sum(img_y[label != 1] == 1)
        tn = np.sum(img_y[label != 1] != 1)
        fn = np.sum(img_y[label == 1] != 1)
        iou1 = tp/(tp+fp+fn)
        presion1 = tp / (tp + fp)
        recall1 = tp / (tp + fn)
        he_iou1 = he_iou1 + iou1
        he_precision1 = he_precision1 + presion1
        he_recall1 = he_recall1 + recall1
        f1score1 = 2 * presion1 * recall1 / (presion1 + recall1+0.00001)

        tp = np.sum(img_y[label == 2] == 2)
        fp = np.sum(img_y[label != 2] == 2)
        tn = np.sum(img_y[label != 2] != 2)
        fn = np.sum(img_y[label == 2] != 2)
        iou2 = tp / (tp + fp + fn)
        presion2 = tp / (tp + fp)
        recall2 = tp / (tp + fn)
        f1score2 = 2 * presion2 * recall2 / (presion2 + recall2+0.00001)
        he_iou2 = he_iou2 + iou2
        he_precision2 = he_precision2 + presion2
        he_recall2 = he_recall2 + recall2

        tp = np.sum(img_y[label == 3] == 3)
        fp = np.sum(img_y[label != 3] == 3)
        tn = np.sum(img_y[label != 3] != 3)
        fn = np.sum(img_y[label == 3] != 3)
        iou3 = tp / (tp + fp + fn)
        presion3 = tp / (tp + fp)
        recall3 = tp / (tp + fn)
        f1score3 = 2 * presion3 * recall3 / (presion3 + recall3+0.00001)
        he_iou3 = he_iou3 + iou3
        he_precision3 = he_precision3 + presion3
        he_recall3 = he_recall3 + recall3

        tp = np.sum(img_y[label == 4] == 4)
        fp = np.sum(img_y[label != 4] == 4)
        tn = np.sum(img_y[label != 4] != 4)
        fn = np.sum(img_y[label == 4] != 4)
        iou4 = tp / (tp + fp + fn)
        presion4 = tp / (tp + fp)
        recall4 = tp / (tp + fn)
        f1score4 = 2 * presion4 * recall4 / (presion4 + recall4+0.00001)
        he_iou4 = he_iou4 + iou4
        he_precision4 = he_precision4 + presion4
        he_recall4 = he_recall4 + recall4

        tp = np.sum(img_y[label == 5] == 5)
        fp = np.sum(img_y[label != 5] == 5)
        tn = np.sum(img_y[label != 5] != 5)
        fn = np.sum(img_y[label == 5] != 5)
        iou5 = tp / (tp + fp + fn)
        presion5 = tp / (tp + fp)
        recall5 = tp / (tp + fn)
        f1score5 = 2 * presion5 * recall5 / (presion5 + recall5+0.00001)
        he_iou5 = he_iou5 + iou5
        he_precision5 = he_precision5 + presion5
        he_recall5 = he_recall5 + recall5

        iou = (iou1+iou2+iou3+iou4+iou5)/5
        presion = (presion1+presion2+presion3+presion4+presion5)/5
        recall = (recall1+recall2+recall3+recall4+recall5)/5
        f1score = (f1score1+f1score2+f1score3+f1score4+f1score5)/5

        iou_he = iou_he + iou
        presion_he = presion_he + presion
        recall_he = recall_he + recall
        f1score_he = f1score_he + f1score

print(time.time()-time1)
mean_accuracy = total_he/n
mean_iou = iou_he/n
mean_presion = presion_he /n
mean_recall = recall_he/n
mean_f1score = f1score_he/n

mean_iou1 = he_iou1/n
mean_precision1 = he_precision1/n
mean_recall1 = he_recall1/n

mean_iou2 = he_iou2/n
mean_precision2 = he_precision2/n
mean_recall2 = he_recall2/n

mean_iou3 = he_iou3/n
mean_precision3 = he_precision3/n
mean_recall3 = he_recall3/n

mean_iou4 = he_iou4/n
mean_precision4 = he_precision4/n
mean_recall4 = he_recall4/n

mean_iou5 = he_iou5/n
mean_precision5 = he_precision5/n
mean_recall5 = he_recall5/n

print('mean acc:', mean_accuracy)
print('mean f1score:', mean_f1score)
print('mean iou:', mean_iou)
print('mean presion:', mean_presion)
print('mean recall:', mean_recall)
print()

print('mean iou1', mean_iou1, 'mean precision1', mean_precision1, 'mean recall1', mean_recall1)
print('mean iou2', mean_iou2, 'mean precision2', mean_precision2, 'mean recall2', mean_recall2)
print('mean iou3', mean_iou3, 'mean precision3', mean_precision3, 'mean recall3', mean_recall3)
print('mean iou4', mean_iou4, 'mean precision4', mean_precision4, 'mean recall4', mean_recall4)
print('mean iou5', mean_iou5, 'mean precision5', mean_precision5, 'mean recall5', mean_recall5)



