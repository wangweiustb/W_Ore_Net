import torch.nn as nn
import torch
from torchvision import models
import torch.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _PSPModule(nn.Module):

    def __init__(self, in_channels):
        super(_PSPModule, self).__init__()
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1, 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample((16, 16))
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(512, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample((16, 16))
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1)
        )

    def forward(self, features):
        pool1 = self.pool1(features)
        pool2 = self.pool2(features)
        pyramids = [features, pool1, pool2]
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


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
        # 这里没有加relu和bn
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
        # 这里没有加relu和bn
        self.doubleconv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.bn_relu(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.doubleconv(x1)
        return x1


class Up2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up2, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel*2, out_channel, kernel_size=2, stride=2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        # 这里没有加relu和bn
        self.doubleconv = DoubleConv(in_channel*3, out_channel)

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


class Unet13sum(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(Unet13sum, self).__init__()
        self.vgg = models.vgg16_bn(True).features
        self.vgg1 = models.vgg16_bn(False).features

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

        self.psp1 = _PSPModule(512)

        self.up_conv6 = Up64_3(1024, 512)
        self.up_conv7 = Up(512, 256)
        self.up_conv8 = Up(256, 128)
        self.up_conv9 = Up(128, 64)

        self.out1 = Out(64, n_class)

        self.middle = nn.Sequential(
            nn.Conv2d(n_class, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.conv21 = nn.Sequential(
            self.vgg1[0],
            self.vgg1[1],
            self.vgg1[2],
            self.vgg1[3],
            self.vgg1[4],
            self.vgg1[5],
        )

        self.conv22 = nn.Sequential(
            self.vgg1[6],
            self.vgg1[7],
            self.vgg1[8],
            self.vgg1[9],
            self.vgg1[10],
            self.vgg1[11],
            self.vgg1[12],
        )

        self.conv23 = nn.Sequential(
            self.vgg1[13],
            self.vgg1[14],
            self.vgg1[15],
            self.vgg1[16],
            self.vgg1[17],
            self.vgg1[18],
            self.vgg1[19],
            self.vgg1[20],
            self.vgg1[21],
            self.vgg1[22],
        )

        self.conv24 = nn.Sequential(
            self.vgg1[23],
            self.vgg1[24],
            self.vgg1[25],
            self.vgg1[26],
            self.vgg1[27],
            self.vgg1[28],
            self.vgg1[29],
            self.vgg1[30],
            self.vgg1[31],
            self.vgg1[32],
        )

        self.conv25 = nn.Sequential(
            self.vgg1[33],
            self.vgg1[34],
            self.vgg1[35],
            self.vgg1[36],
            self.vgg1[37],
            self.vgg1[38],
            self.vgg1[39],
            self.vgg1[40],
            self.vgg1[41],
            self.vgg1[42],
        )
        self.psp2 = _PSPModule(512)

        self.up_conv26 = Up64_3(1024, 512)
        self.up_conv27 = Up(512, 256)
        self.up_conv28 = Up(256, 128)
        self.up_conv29 = Up(128, 64)

        self.out2 = Out(64, n_class)

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.psp1(x5)

        x6 = self.up_conv6(x5, x4)
        x7 = self.up_conv7(x6, x3)
        x8 = self.up_conv8(x7, x2)
        x9 = self.up_conv9(x8, x1)
        out1 = self.out1(x9)

        xmiddle = self.middle(out1)

        x21 = self.conv21(xmiddle)
        x21 = x21 + x1

        x22 = self.conv22(x21)
        x22 = x22 + x2

        x23 = self.conv23(x22)
        x23 = x23 + x3

        x24 = self.conv24(x23)
        x24 = x24 + x4

        x25 = self.conv25(x24)
        x25 = self.psp2(x25)
        x25 = x25 + x5

        x26 = self.up_conv26(x25, x24)
        x27 = self.up_conv27(x26, x23)
        x28 = self.up_conv28(x27, x22)
        x29 = self.up_conv29(x28, x21)

        out2 = self.out2(x29)

        return out1, out2

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


model = Unet13sum(3, 6)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_num, trainable_num)


