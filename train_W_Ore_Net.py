import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torchvision.transforms import transforms

from model.W_Ore_Net import Unet13sum
import cv2 as cv
import os
import time
import numpy as np
import random
from tensorboardX import SummaryWriter
from PIL import Image


seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


time1 = time.time()
path_val_acc = './pth/1acc.pth'
path_val_iou = './pth/1iou.pth'
writer = SummaryWriter('runs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 2
num_epochs = 150
train_img_path = 'train/img256'
train_label_path = 'train/label256'

val_img_path = './test/img256'
val_label_path = './test/label256'

# 训练集path
n = len(os.listdir(val_img_path))
val_imgs = []  # 训练集图片与标签的地址 是一个列表，列表中的每个元素是一个元组，每个地址是一个字符串

for i in range(n):
    img = os.path.join(val_img_path, "%d.png" % i)
    label = os.path.join(val_label_path, "%d.png" % i)
    val_imgs.append((img, label))

n = len(os.listdir(train_img_path))
train_imgs = []  # 训练集图片与标签的地址 是一个列表，列表中的每个元素是一个元组，每个地址是一个字符串

for i in range(n):
    img = os.path.join(train_img_path, "%d.png" % i)
    label = os.path.join(train_label_path, "%d.png" % i)
    train_imgs.append((img, label))


class MyTrainDatas(Dataset):
    def __init__(self, train_img):
        self.train_img = train_img

        self.data_transforms = {
            'imgs': transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ]),
            'masks': transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
                transforms.RandomRotation(10),
                # transforms.ToTensor()
            ]),
        }

    def __getitem__(self, index):
        x_path, y_path = self.train_img[index]
        img = Image.open(x_path)
        gt = Image.open(y_path)

        seeds = np.random.randint(12345)
        random.seed(seeds)
        img = self.data_transforms['imgs'](img)
        random.seed(seeds)
        gt = self.data_transforms['masks'](gt)
        gt = np.array(gt, dtype='int64')
        gt = torch.from_numpy(gt)
        # gt = torch.squeeze(gt).long()
        return img, gt

    def __len__(self):
        return len(self.train_img)


class MyValDatas(Dataset):
    def __init__(self, train_img):
        self.train_img = train_img

        self.data_transforms = {
            'imgs': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'masks': transforms.Compose([
                transforms.ToTensor()
            ])
        }

    def __getitem__(self, index):
        x_path, y_path = self.train_img[index]
        img = Image.open(x_path)
        gt = Image.open(y_path)
        img = self.data_transforms['imgs'](img)
        gt = np.array(gt, dtype='int64')
        gt = torch.from_numpy(gt)
        return img, gt

    def __len__(self):
        return len(self.train_img)


val_predict_path = './test/img256'
label_path2 = './test/label256'


img_path = []  # 预测图片的地址
label_path = []
n = len(os.listdir(val_predict_path))
for j in range(n):
    img1 = os.path.join(val_predict_path, "%d.png" % j)
    label1 = os.path.join(label_path2, '%d.png' % j)
    img_path.append(img1)
    label_path.append(label1)


train_dataset = MyTrainDatas(train_imgs)
print('len train_dataset', len(train_dataset))
train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = MyValDatas(val_imgs)
print('len val_dataset', len(val_dataset))
val_dataloaders = DataLoader(val_dataset, batch_size=batch_size)

model = Unet13sum(3, 6)

if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = nn.DataParallel(model)
model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

min_loss = 1000
best_train_accuracy = 0.0
best_train_accuracy_epoch = 0
best_val_accuracy = 0.0
best_val_accuracy_epoch = 0
best_val_loss = 1000.0
best_val_loss_epoch = 0

last_loss_ce = 0
last_loss_dice = 0

best_mean_iou = 0.0
best_mean_iou_epoch = 0
best_mean_acc = 0.0


for epoch in range(num_epochs):
    time2 = time.time()
    model.train()
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    epoch_loss = 0.0
    for step, data in enumerate(train_dataloaders):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs)
        labels_for_ce = torch.squeeze(labels, dim=1).long()

        loss1 = criterion(outputs1, labels_for_ce)
        loss2 = criterion(outputs2, labels_for_ce)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print("%d/%d,train_loss:%0.4f" % (step + 1, 386, loss.item()))  # 这里需要改变批次
    # scheduler.step()
    print("epoch: %d epoch_train_mean_loss:%0.4f" % (epoch + 1, epoch_loss / 772))  # 这里需要改变批次
    writer.add_scalar("loss_epoch", epoch_loss / 772, epoch + 1)  # 这里需要改变批次

    # 每个epoch统计的total值必在epoch之下，批次之上
    model.eval()
    train_total_accuracy = 0.0
    val_total_accuracy = 0.0
    val_best_epoch = 0
    val_loss = 0.0

    total_he = 0.0
    iou_he = 0.0

    with torch.no_grad():
        for i in range(n):
            img = Image.open(os.path.join(val_predict_path, '%d.png' % i))
            label = Image.open(os.path.join(label_path2, '%d.png' % i))
            label = np.array(label, dtype='int64')

            img_predict = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(img)

            img_predict = torch.unsqueeze(img_predict, 0)
            _, y = model(img_predict.to(device))  # 单loss
            y = torch.argmax(y, 1)

            y = y.cpu().detach()
            img_y = torch.squeeze(y).numpy()

            he = np.zeros((256, 256))
            he[img_y == label] = 1
            he = he.sum()
            mean_he = he / (256 * 256)
            #
            total_he = total_he + mean_he

            tp = np.sum(img_y[label == 1] == 1)
            fp = np.sum(img_y[label != 1] == 1)
            tn = np.sum(img_y[label != 1] != 1)
            fn = np.sum(img_y[label == 1] != 1)
            iou1 = tp / (tp + fp + fn)

            tp = np.sum(img_y[label == 2] == 2)
            fp = np.sum(img_y[label != 2] == 2)
            tn = np.sum(img_y[label != 2] != 2)
            fn = np.sum(img_y[label == 2] != 2)
            iou2 = tp / (tp + fp + fn)

            tp = np.sum(img_y[label == 3] == 3)
            fp = np.sum(img_y[label != 3] == 3)
            tn = np.sum(img_y[label != 3] != 3)
            fn = np.sum(img_y[label == 3] != 3)
            iou3 = tp / (tp + fp + fn)

            tp = np.sum(img_y[label == 4] == 4)
            fp = np.sum(img_y[label != 4] == 4)
            tn = np.sum(img_y[label != 4] != 4)
            fn = np.sum(img_y[label == 4] != 4)
            iou4 = tp / (tp + fp + fn)

            tp = np.sum(img_y[label == 5] == 5)
            fp = np.sum(img_y[label != 5] == 5)
            tn = np.sum(img_y[label != 5] != 5)
            fn = np.sum(img_y[label == 5] != 5)
            iou5 = tp / (tp + fp + fn)

            iou = (iou1 + iou2 + iou3 + iou4 + iou5) / 5

            iou_he = iou_he + iou

    mean_accuracy = total_he / n
    mean_iou = iou_he / n
    print('epoch: %d, mean_val_acc:  %.4f' % (epoch + 1, mean_accuracy))
    print('epoch: %d, mean_val_iou:  %.4f' % (epoch + 1, mean_iou))
    writer.add_scalar("val_acc_epoch", mean_accuracy, epoch + 1)
    writer.add_scalar("val_iou_epoch", mean_iou, epoch + 1)
    # if mean_accuracy > best_mean_acc:
    #     best_mean_acc = mean_accuracy
    #     best_val_accuracy_epoch = epoch
    #     torch.save(model.state_dict(), path_val_acc)

    if mean_iou > best_mean_iou:
        best_mean_iou = mean_iou
        best_mean_iou_epoch = epoch
        torch.save(model.state_dict(), path_val_iou)

    print('epoch time', time.time() - time2)
writer.close()


print('best val_acc：%.4f' % best_mean_acc)
print('best val accuracy epoch：%d' % best_val_accuracy_epoch)
print('best val_iou：%.4f' % best_mean_iou)
print('best val iou epoch：%d' % best_mean_iou_epoch)

print('total trainning time', time.time() - time1)
print("finised trainning")

# tensorboard --logdir=runs --port 6008


