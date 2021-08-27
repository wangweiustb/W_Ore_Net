import torch.nn as nn
import torch
from torchvision import models
import torch.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv_bn_re = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            # nn.Dropout(drop),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_bn_re(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.doubleconv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.bn_relu(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.doubleconv(x1)
        return x1


class Up64_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up64_3, self).__init__()
        self.up = nn.ConvTranspose2d(512, out_channel, kernel_size=2, stride=2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.doubleconv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.bn_relu(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.doubleconv(x1)
        return x1


class Out(nn.Module):
    def __init__(self, in_channel, n_class):
        super(Out, self).__init__()
        self.out = nn.Conv2d(in_channel, n_class, 1)

    def forward(self, x):
        return self.out(x)


class Unet(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(Unet, self).__init__()
        self.vgg = models.vgg16_bn(False).features

        self.conv1 = nn.Sequential(
            self.vgg[0],
            self.vgg[1],
            self.vgg[2],
            self.vgg[3],
            self.vgg[4],
            self.vgg[5],
        )

        self.conv2 = nn.Sequential(
            self.vgg[6],
            self.vgg[7],
            self.vgg[8],
            self.vgg[9],
            self.vgg[10],
            self.vgg[11],
            self.vgg[12],
        )

        self.conv3 = nn.Sequential(
            self.vgg[13],
            self.vgg[14],
            self.vgg[15],
            self.vgg[16],
            self.vgg[17],
            self.vgg[18],
            self.vgg[19],
            self.vgg[20],
            self.vgg[21],
            self.vgg[22],
        )

        self.conv4 = nn.Sequential(
            self.vgg[23],
            self.vgg[24],
            self.vgg[25],
            self.vgg[26],
            self.vgg[27],
            self.vgg[28],
            self.vgg[29],
            self.vgg[30],
            self.vgg[31],
            self.vgg[32],
        )

        self.conv5 = nn.Sequential(
            self.vgg[33],
            self.vgg[34],
            self.vgg[35],
            self.vgg[36],
            self.vgg[37],
            self.vgg[38],
            self.vgg[39],
            self.vgg[40],
            self.vgg[41],
            self.vgg[42],
        )

        self.up_conv6 = Up64_3(1024, 512)
        self.up_conv7 = Up(512, 256)
        self.up_conv8 = Up(256, 128)
        self.up_conv9 = Up(128, 64)

        self.out = Out(64, n_class)

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        # x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.up_conv6(x5, x4)
        x7 = self.up_conv7(x6, x3)
        x8 = self.up_conv8(x7, x2)
        x9 = self.up_conv9(x8, x1)
        out = self.out(x9)
        return out


model = Unet(3, 6)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_num, trainable_num)


