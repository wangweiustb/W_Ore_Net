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


class Up_att(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_att, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.bn_relu(x1)
        return x1


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv = DoubleConv(4*F_int, 2*F_int)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        out = torch.cat([g, out], dim=1)
        out = self.conv(out)
        return out


class Out(nn.Module):
    def __init__(self, in_channel, n_class):
        super(Out, self).__init__()
        self.out = nn.Conv2d(in_channel, n_class, 1)

    def forward(self, x):
        return self.out(x)


class Unetatt(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(Unetatt, self).__init__()
        self.conv1 = DoubleConv(in_channel, 64)
        self.conv2 = Down(64, 128)
        self.conv3 = Down(128, 256)
        self.conv4 = Down(256, 512)
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.up_conv6 = Up_att(1024, 512)
        self.att6 = Attention_block(512, 512, 256)
        self.up_conv7 = Up_att(512, 256)
        self.att7 = Attention_block(256, 256, 128)
        self.up_conv8 = Up_att(256, 128)
        self.att8 = Attention_block(128, 128, 64)
        self.up_conv9 = Up_att(128, 64)
        self.att9 = Attention_block(64, 64, 32)
        self.out = Out(64, n_class)

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.up_conv6(x5)
        x6 = self.att6(x6, x4)
        x7 = self.up_conv7(x6)
        x7 = self.att7(x7, x3)
        x8 = self.up_conv8(x7)
        x8 = self.att8(x8, x2)
        x9 = self.up_conv9(x8)
        x9 = self.att9(x9, x1)
        out = self.out(x9)
        return out

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


class FCN8(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(FCN8, self).__init__()
        self.conv1 = DoubleConv(in_channel, 64)
        self.down_conv2 = Down(64, 128)
        self.down_conv3 = Down(128, 256)
        self.down_conv4 = Down(256, 512)
        self.down_drop5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            )
        self.up_conv6 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.up_conv7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 6, 1, 1),
        )

        self.out = nn.Upsample((256, 256))

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_drop5(x4)
        x6 = self.up_conv6(x5)
        x6 = x6 + x4
        x7 = self.up_conv7(x6)
        x7 = x7 + x3
        x8 = self.conv8(x7)
        out = self.out(x8)

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


class VGGFCN(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(VGGFCN, self).__init__()
        self.vgg = models.vgg16_bn(True).features

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

        self.up_conv6 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.up_conv7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 6, 1, 1),
        )

        self.out = nn.Upsample((256, 256))

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.up_conv6(x5)
        x6 = x6 + x4
        x7 = self.up_conv7(x6)
        x7 = x7 + x3
        x8 = self.conv8(x7)
        out = self.out(x8)

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


class _PSPModule(nn.Module):
    def __init__(self, in_channels):
        super(_PSPModule, self).__init__()
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1, 1),
            nn.ReLU(),
            nn.Upsample((16, 16))
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample((16, 16))
        )
        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample((16, 16))
        )
        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample((16, 16))
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        pool1 = self.pool1(features)
        pool2 = self.pool2(features)
        pool3 = self.pool3(features)
        pool4 = self.pool4(features)
        pyramids = [features, pool1, pool2, pool3, pool4]
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class Res50PSP(nn.Module):
    def __init__(self, in_channel, n_class, _init_weight=False):
        super(Res50PSP, self).__init__()
        self.resnet = models.resnet50(True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4

        self.psp = _PSPModule(2048)
        self.out1 = nn.Sequential(
            nn.Conv2d(2048, 6, 1, 1, 1),
        )
        self.out = nn.Upsample((256, 256))

        if _init_weight:
            self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.psp(x5)
        x6 = self.out1(x5)
        out = self.out(x6)

        return out


model = Res50PSP(3, 6)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_num, trainable_num)


