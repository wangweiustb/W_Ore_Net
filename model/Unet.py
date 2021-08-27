import torch.nn as nn
import torch
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv_bn_re = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
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


class Out(nn.Module):
    def __init__(self, in_channel, n_class):
        super(Out, self).__init__()
        self.out = nn.Conv2d(in_channel, n_class, 1)

    def forward(self, x):
        return self.out(x)


class Unet(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_channel, 64)
        self.down_conv2 = Down(64, 128)
        self.down_conv3 = Down(128, 256)
        self.down_conv4 = Down(256, 512)
        self.down_conv5 = Down(512, 1024)
        # self.down_drop5 = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(512, 1024, 3, 1, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Conv2d(1024, 1024, 3, 1, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5)
        #     )
        self.up_conv6 = Up(1024, 512)
        self.up_conv7 = Up(512, 256)
        self.up_conv8 = Up(256, 128)
        self.up_conv9 = Up(128, 64)
        self.out = Out(64, n_class)

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)
        # x5 = self.down_drop5(x4)

        x6 = self.up_conv6(x5, x4)
        x7 = self.up_conv7(x6, x3)
        x8 = self.up_conv8(x7, x2)
        x9 = self.up_conv9(x8, x1)
        out = self.out(x9)

        return out

    # 初始化需要改进，先不初始化试试
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


model = Unet(3, 6)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_num, trainable_num)




