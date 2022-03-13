import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data


class conv_block(nn.Module):
    """
    Convolution Block 
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Res_Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch = 4):
        super(Res_Unet, self).__init__()
        n1 = 64
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, n1)
        self.Conv2 = conv_block(n1, n1)
        self.Conv3 = conv_block(n1, n1)
        self.Conv4 = conv_block(n1, n1)
        self.Conv5 = conv_block(n1, n1)

        self.Up5 = up_conv(n1, n1)
        self.Up_conv5 = conv_block(n1*2, n1)

        self.Up4 = up_conv(n1, n1)
        self.Up_conv4 = conv_block(n1*2, n1)

        self.Up3 = up_conv(n1, n1)
        self.Up_conv3 = conv_block(n1*2, n1)

        self.Up2 = up_conv(n1, n1)
        self.Up_conv2 = conv_block(n1*2, n1)

        self.Conv = nn.Conv2d(n1, out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)  #  224*224*64

        e2 = self.Maxpool1(e1)  #  112*112*64
        e2 = self.Conv2(e2)     #  112*112*64

        e3 = self.Maxpool2(e2)  #  56*56*64
        e3 = self.Conv3(e3)  #  56*56*64

        e4 = self.Maxpool3(e3)  # 28*28*64
        e4 = self.Conv4(e4)  # 28*28*64

        e5 = self.Maxpool4(e4)  # 14*14*64
        e5 = self.Conv5(e5)  # 14*14*64

        d5 = self.Up5(e5)  # 28*28*64
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)  #  56*56*64
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)  #  112*112*64
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)  #  224*224*64
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out_1 = self.Conv(d2)

        out = self.active(out_1)

        return out

