import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
from torch.autograd import Variable
import argparse
import geopandas as gpd
from sklearn.metrics import roc_curve, auc  # 計算roc和auc
import pix2pix_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

seed = 164
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

SourcePath = pix2pix_data.SourcePath
Path_QPE = pix2pix_data.Path_QPE
Path_Pix2pix = pix2pix_data.Path_Pix2pix
Path_shp = pix2pix_data.Path_shp
Region = pix2pix_data.Region
useWD = True
log2_pre = True


class NNDataset(Dataset):
    def __init__(self, X, Y, Test=False):
        self.X = X
        self.Y = Y
        self.Path = Path_QPE
        print(len(self.Y))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        X_ = [None] * len(self.X[i])
        X_time = [None] * len(self.X[i])
        for c in range(len(self.X[i])):
            X_[c] = self.Preprocess(np.load(self.X[i][c]), self.X[i][c])
            X_time[c] = self.X[i][c]

        X_ = np.concatenate(X_, axis=2)
        Y_ = self.Y_Preprocess(self.Y[i][0])

        if self.Y[i][1] == 1:
            X_ = X_[::-1, :, :].copy()
            Y_ = Y_[::-1, :].copy()
        if self.Y[i][1] == 2:
            X_ = X_[:, ::-1, :].copy()
            Y_ = Y_[:, ::-1].copy()
        if self.Y[i][1] == 3:
            X_ = X_[::-1, ::-1, :].copy()
            Y_ = Y_[::-1, ::-1].copy()

        return torch.from_numpy(X_).float(), torch.from_numpy(Y_), X_time, self.Y[i][0]
        # return torch.from_numpy(X_).float(), torch.from_numpy(Y_).float().unsqueeze(0), X_time, self.Y[i][0]

    def Preprocess(self, array, ci):
        # wissdom_out_Taiwan_mosaic_202105311730.npy
        if ci[-42:-17] == "wissdom_out_Taiwan_mosaic":
            return array[:, :, ::-1]
        return array

    def Y_Preprocess(self, Y_path):
        if not os.path.exists(Y_path):
            return np.zeros(1)
        array = np.load(Y_path)
        array = pix2pix_data.SplitRegion(array, "TW", Region)
        if Region == "RS_TW":
            h, w = pix2pix_data.Get_LongLat(Region, True)
            array = cv2.resize(array, (w, h))
        if log2_pre:
            array = np.log2(array + 1) / np.log2(pix2pix_data.MaxPre + 1)
        else:
            array = array / pix2pix_data.MaxPre
        return array


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.mean((outputs - targets) ** 2 * (targets + 1))


# UNet --------------------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# UNet ＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=24 + 48):
        super(AutoEncoder, self).__init__()

        self.in1 = nn.Conv2d(in_channels, 36, 5, 1, 2)
        self.in2 = nn.Conv2d(36, 18, 5, 1, 2)
        self.in3 = nn.Conv2d(18, 8, 5, 1, 2)
        self.in4 = nn.Conv2d(8, 1, 5, 1, 2)

        self.out1 = nn.Conv2d(1, 8, 5, 1, 2)
        self.out2 = nn.Conv2d(8, 18, 5, 1, 2)
        self.out3 = nn.Conv2d(18, 36, 5, 1, 2)
        self.out4 = nn.Conv2d(36, in_channels, 5, 1, 2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        x = F.relu(self.in1(x))
        x = F.relu(self.in2(x))
        x = F.relu(self.in3(x))
        x = F.relu(self.in4(x))

        hid = x

        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        x = F.relu(self.out4(x))

        x = x.permute(0, 2, 3, 1)
        batch = x.shape[0]
        channel = x.shape[3]
        height = x.shape[1]
        width = x.shape[2]

        return x, hid


# GAN --------------------------------------------------
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet2(nn.Module):
    def __init__(self, in_channels=24, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.conv_hPA0 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA0_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA1_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA2_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA3 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv_hPA3_2 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv_hPA3_3 = nn.Conv2d(48, 24, 3, padding=1)

        # self.down1 = UNetDown(24, 64, normalize=False)
        self.down1 = UNetDown(24*2, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(256 + 256, 128)
        self.up5 = UNetUp(128 + 128, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        x = x.permute(0, 3, 1, 2)
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        x0 = F.relu(self.conv_hPA0(x[:, 0:8, :, :]))
        x1 = F.relu(self.conv_hPA1(x[:, 8:16, :, :]))
        x2 = F.relu(self.conv_hPA2(x[:, 16:24, :, :]))
        x0 = F.relu(self.conv_hPA0_2(x0))
        x1 = F.relu(self.conv_hPA1_2(x1))
        x2 = F.relu(self.conv_hPA2_2(x2))
        x_cat = torch.cat((x0, x1, x2), 1)  # torch.Size([7, 24, 64, 64])
        if channel == 24 + 48:
            # x3 = x[:, 24:, :, :].clone()
            x3 = F.relu(self.conv_hPA3(x[:, 24:, :, :]))
            x3 = F.relu(self.conv_hPA3_2(x3))
            # x3 = F.tanh(self.conv_hPA3_3(x3))
            # x_cat = F.relu(x_cat + x3)
            x3 = F.relu(self.conv_hPA3_3(x3))
            x_cat = torch.cat((x_cat, x3), 1)  # torch.Size([7, 24, 64, 64])

        d1 = self.down1(x_cat)  # torch.Size([7, 64, 32, 32])
        d2 = self.down2(d1)  # torch.Size([7, 128, 16, 16])
        d3 = self.down3(d2)  # torch.Size([7, 256, 8, 8])
        d4 = self.down4(d3)  # torch.Size([7, 512, 4, 4])
        d5 = self.down5(d4)  # torch.Size([7, 512, 2, 2])
        d6 = self.down6(d5)  # torch.Size([7, 512, 1, 1])
        u1 = self.up1(d6, d5)  # torch.Size([7, 1024, 2, 2])
        u2 = self.up2(u1, d4)  # torch.Size([7, 1024, 4, 4])
        u3 = self.up3(u2, d3)  # torch.Size([7, 512, 8, 8])
        u4 = self.up4(u3, d2)  # torch.Size([7, 256, 16, 16])
        u5 = self.up5(u4, d1)  # torch.Size([7, 128, 32, 32])
        fin = self.final(u5)  # torch.Size([7, 1, 64, 64])

        return fin


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=24, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.conv_hPA0 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA0_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA1_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA2_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_hPA3 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv_hPA3_2 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv_hPA3_3 = nn.Conv2d(48, 24, 3, padding=1)

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(256 + 256, 128)
        self.up5 = UNetUp(128 + 128, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        x = x.permute(0, 3, 1, 2)
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        x0 = F.relu(self.conv_hPA0(x[:, 0:8, :, :]))
        x1 = F.relu(self.conv_hPA1(x[:, 8:16, :, :]))
        x2 = F.relu(self.conv_hPA2(x[:, 16:24, :, :]))
        x0 = F.relu(self.conv_hPA0_2(x0))
        x1 = F.relu(self.conv_hPA1_2(x1))
        x2 = F.relu(self.conv_hPA2_2(x2))
        x_cat = torch.cat((x0, x1, x2), 1)
        if channel == 24 + 48:
            # x3 = x[:, 24:, :, :].clone()
            x3 = F.relu(self.conv_hPA3(x[:, 24:, :, :]))
            x3 = F.relu(self.conv_hPA3_2(x3))
            x3 = F.tanh(self.conv_hPA3_3(x3))
            x_cat = F.relu(x_cat + x3)

        d1 = self.down1(x_cat)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        fin = self.final(u5)

        return fin


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # (1, 2, 64, 64)
            *discriminator_block(1 + 1, 64, normalization=False),
            # *discriminator_block(24 + 1, 64, normalization=False),
            # *discriminator_block(1, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            # (1, 1, 4, 4)
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

        # x = img_A.permute(0, 3, 1, 2)
        # batch = x.shape[0]
        # channel = x.shape[1]
        # height = x.shape[2]
        # width = x.shape[3]
        # if channel == 24 + 48:
        #     x = x[:, :24, :, :]
        # img_input = torch.cat((x, img_B), 1)

        # img_input = img_B
        return self.model(img_input)


# GAN ＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾


class pix2pixModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.generator = None
        self.discriminator = None
        self.autoencoder = None
        self.unet = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.optimizer_A = None
        self.train_D = True
        self.lr_G = opt.lr
        self.lr_D = opt.lr/20
        # self.loss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.customLoss = customLoss()
        self.MSELoss = nn.MSELoss()
        self.cmap = None
        self.norm = None
        self.bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        self.epoch_num = opt.epoch_num
        self.threshold = opt.CSI_Threshold
        self.roc_p = None
        self.roc_gt = None
        self.highmse = None
        self.heavymse = None
        self.mbe = None
        self.generator_Load_Unet = False  # 將unet.pth load進self.generator, for test使用
        self.Unet_replace_generator = False  # 將UNet load進self.generator, for train使用
        if os.path.exists(os.path.join(SourcePath, "site.npy")):
            self.site = pix2pix_data.SplitRegion(np.load(os.path.join(SourcePath, "site.npy")), "TW", Region)
        else:
            self.site = None

    def train(self, DataLoader_train, DataLoader_val):
        self.load_model(False)
        self.load_autoencoder(True)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=(0.5, 0.999))

        lessLoss_G, lessLoss_D = None, None
        count_G, count_D = 0, 0
        log_train, log_val, log_mse = [], [], []
        log_loss_D_train, log_loss_GAN_train, log_loss_D_val, log_loss_GAN_val = [], [], [], []
        for epoch in range(1, self.epoch_num + 1):
            NowTime = time()
            loss_D_train, loss_GAN_train, loss_pixel_train = 0, 0, 0

            for i, (train_x, train_y, _, _) in enumerate(DataLoader_train, 1):
                train_y = train_y.float().unsqueeze(1)
                self.generator.train()
                self.discriminator.train()
                loss_D, _, _, loss_pixel, loss_GAN, _ = self.train_step(train_x, train_y)
                loss_D_train += loss_D
                loss_GAN_train += loss_GAN
                loss_pixel_train += loss_pixel

            if epoch % 1 == 0:
                # val --------------------------------------------------
                self.generator.eval()
                self.discriminator.eval()
                with torch.no_grad():
                    loss_D_val, loss_GAN_val, loss_pixel_val, loss_pred_real, loss_pred_fake, mse_val = 0, 0, 0, 0, 0, 0
                    for _, (val_x, val_y, _, YName) in enumerate(DataLoader_val):
                        val_y = val_y.float().unsqueeze(1)
                        loss_D, pred_real, pred_fake, loss_pixel, loss_GAN, output = self.train_step(val_x, val_y, train=False)
                        loss_D_val += loss_D
                        loss_pixel_val += loss_pixel
                        loss_GAN_val += loss_GAN
                        loss_pred_real += pred_real
                        loss_pred_fake += pred_fake

                        for b in range(val_x.shape[0]):
                            p = output[b, :, :].cpu().numpy().reshape(self.opt.model_h, self.opt.model_w)
                            if log2_pre:
                                p = 2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1
                            else:
                                p *= pix2pix_data.MaxPre
                            if Region == "RS_TW":
                                p = cv2.resize(p, (self.opt.img_w, self.opt.img_h))
                            if os.path.exists(YName[b]):
                                gt = np.load(YName[b])
                                gt = pix2pix_data.SplitRegion(gt, "TW", Region)
                                site = self.site
                                gt, p = gt[site == 1], p[site == 1]
                                mse = np.square(np.subtract(gt, p)).mean()
                                mse_val += mse
                # val --------------------------------------------------

                loss_pixel_train *= 1000
                loss_pixel_val *= 1000
                print(
                    "[Epoch %d, duration=%.2f] [train D loss: %f, GAN loss: %f, pixel loss: %f] "
                    "[val D loss: %f, GAN loss: %f, pixel loss: %f, pred_real: %f, pred_fake: %f] [mse: %f]"
                    % (
                        epoch, (time() - NowTime) / 60,
                        loss_D_train / len(DataLoader_train), loss_GAN_train / len(DataLoader_train),
                        loss_pixel_train / len(DataLoader_train),
                        loss_D_val / len(DataLoader_val), loss_GAN_val / len(DataLoader_val),
                        loss_pixel_val / len(DataLoader_val),
                        loss_pred_real / len(DataLoader_val), loss_pred_fake / len(DataLoader_val),
                        mse_val / len(DataLoader_val)
                    )
                )

                # if lessLoss_D is None or loss_D_val < lessLoss_D:
                #     lessLoss_D = loss_D_val
                #     count_D = 0
                # else:
                #     count_D += 1
                #     if count_D == 1 or count_D == 6:
                #         self.lr_D *= 0.9
                #         self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D)

                if lessLoss_G is None or mse_val < lessLoss_G:
                    lessLoss_G = mse_val
                    count_G = 0
                    self.save_model()
                else:
                    count_G += 1
                    if count_G == 1 or count_G == 6:
                        self.lr_G *= 0.9
                        print("lr = ", self.lr_G)
                        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G)
                        #
                        self.lr_D *= 0.9
                        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D)

                # 紀錄 loss --------------------------------------------------
                log_train.append(loss_pixel_train / len(DataLoader_train))
                log_val.append(loss_pixel_val / len(DataLoader_val))
                try:
                    plt.plot(log_train, label="train")
                    plt.plot(log_val, label="val")
                    plt.legend()
                    plt.xlabel("epoch")
                    half = log_train[int(len(log_train) / 2):] + log_val[int(len(log_val) / 2):]
                    plt.ylim(min(half), 2 * max(half) - min(half))
                    plt.ylabel("loss(10^-3)")
                    plt.grid(linestyle="--", alpha=0.3)
                    plt.savefig(Path_Pix2pix + "/loss.png")
                    plt.close()
                except:
                    plt.close()

                log_loss_D_train.append(loss_D_train / len(DataLoader_train))
                log_loss_GAN_train.append(loss_GAN_train / len(DataLoader_train))
                log_loss_D_val.append(loss_D_val / len(DataLoader_val))
                log_loss_GAN_val.append(loss_GAN_val / len(DataLoader_val))
                try:
                    plt.plot(log_loss_D_train, label="D_Loss_train")
                    plt.plot(log_loss_GAN_train, label="G_Loss_train")
                    plt.plot(log_loss_D_val, label="D_Loss_val")
                    plt.plot(log_loss_GAN_val, label="G_Loss_val")
                    plt.legend()
                    plt.xlabel("epoch")
                    plt.ylabel("loss")
                    plt.grid(linestyle="--", alpha=0.3)
                    plt.savefig(Path_Pix2pix + "/loss_gan.png")
                    plt.close()
                except:
                    plt.close()
                # 紀錄 loss ＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾＾

                if count_G > 15:
                    print("break epoch = ", epoch)
                    break
                # if count_D > 20 and self.train_D:
                #     self.train_D = False
                #     print("stop Train Discriminator")

    def trainAuto(self, DataLoader_train, DataLoader_val):
        self.load_autoencoder(False)
        lr = 0.0001
        self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

        lessLoss = None
        count = 0
        log_train, log_val = [], []
        epoch_num = self.epoch_num + 9999
        epoch_num = 2
        for epoch in range(1, epoch_num):
            NowTime = time()
            loss_train = 0

            for _, (train_x, _, _, _) in enumerate(DataLoader_train, 1):
                self.autoencoder.train()
                self.optimizer_A.zero_grad()
                x, _ = self.autoencoder(train_x)
                train_x = train_x.to(device)
                loss = self.MSELoss(train_x, x)
                loss.backward()
                self.optimizer_A.step()

                loss_train += loss.item()

            if epoch % 1 == 0:
                self.autoencoder.eval()
                with torch.no_grad():
                    loss_val = 0
                    for _, (val_x, _, _, _) in enumerate(DataLoader_val):
                        x, _ = self.autoencoder(val_x)
                        val_x = val_x.to(device)
                        loss = self.MSELoss(val_x, x)
                        loss_val += loss.item()

                print(
                    "[Epoch %d, duration=%.2f] [train loss: %f] "
                    "[val loss: %f]"
                    % (
                        epoch, (time() - NowTime) / 60,
                        loss_train / len(DataLoader_train), loss_val / len(DataLoader_val)
                    )
                )
                if lessLoss is None or loss_val < lessLoss:
                    lessLoss = loss_val
                    count = 0
                    torch.save(self.autoencoder.module.state_dict(), Path_Pix2pix + "/autoencoder.pth")
                else:
                    count += 1
                    # if count == 1 or count == 6 or count == 11:
                    #     lr *= 0.9
                    #     print("lr = ", lr)
                    #     self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
                # 紀錄 loss
                log_train.append(loss_train / len(DataLoader_train))
                log_val.append(loss_val / len(DataLoader_val))
                try:
                    plt.plot(log_train, label="train")
                    plt.plot(log_val, label="val")
                    plt.legend()
                    plt.xlabel("epoch")
                    half = log_train[int(len(log_train) / 2):] + log_val[int(len(log_val) / 2):]
                    plt.ylim(min(half), 2 * max(half) - min(half))
                    plt.ylabel("loss")
                    plt.grid(linestyle="--", alpha=0.3)
                    plt.savefig(Path_Pix2pix + "/auto_loss.png")
                    plt.close()
                except:
                    plt.close()

                if count > 15:
                    print("break epoch = ", epoch)
                    break

    def trainUNet(self, DataLoader_train, DataLoader_val):
        self.load_unet(False)
        lr = 0.00001
        self.optimizer_A = torch.optim.Adam(self.unet.parameters(), lr=lr)

        lessLoss = None
        count = 0
        log_train, log_val, log_mse = [], [], []
        for epoch in range(1, self.epoch_num + 9999):
            NowTime = time()
            loss_train = 0

            for i, (train_x, train_y, _, _) in enumerate(DataLoader_train, 1):
                train_y = train_y.float().unsqueeze(1)
                self.unet.train()
                # if train_x.shape[3] == 24 + 48:  # frozen
                #     None
                # elif train_x.shape[3] == 24:
                #     continue
                self.optimizer_A.zero_grad()
                outputs = self.unet(train_x)
                targets = train_y.to(device)
                loss = self.MSELoss(outputs, targets)
                loss.backward()
                self.optimizer_A.step()

                loss_train += loss.item()

            if epoch % 1 == 0:
                # val --------------------------------------------------
                self.unet.eval()
                with torch.no_grad():
                    loss_val = 0
                    mse_val = 0
                    for _, (val_x, val_y, _, YName) in enumerate(DataLoader_val):
                        val_y = val_y.float().unsqueeze(1)
                        output = self.unet(val_x)
                        targets = val_y.to(device)
                        loss = self.customLoss(output, targets)
                        loss_val += loss.item()

                        for b in range(val_x.shape[0]):
                            p = output[b, :, :].cpu().numpy().reshape(self.opt.model_h, self.opt.model_w)
                            if log2_pre:
                                p = 2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1
                            else:
                                p *= pix2pix_data.MaxPre
                            if Region == "RS_TW":
                                p = cv2.resize(p, (self.opt.img_w, self.opt.img_h))
                            if os.path.exists(YName[b]):
                                gt = np.load(YName[b])
                                gt = pix2pix_data.SplitRegion(gt, "TW", Region)
                                site = self.site
                                gt, p = gt[site == 1], p[site == 1]
                                mse = np.square(np.subtract(gt, p)).mean()
                                mse_val += mse
                # val --------------------------------------------------

                print(
                    "[Epoch %d, duration=%.2f] [train loss: %f] [val loss: %f] [mse: %f]"
                    % (
                        epoch, (time() - NowTime) / 60,
                        loss_train / len(DataLoader_train), loss_val / len(DataLoader_val),
                        mse_val / len(DataLoader_val)
                    )
                )
                if lessLoss is None or mse_val < lessLoss:
                    lessLoss = mse_val
                    count = 0
                    self.save_unet()
                else:
                    count += 1
                    # if count == 1 or count == 6:
                    #     lr *= 0.9
                    #     print("lr = ", lr)
                    #     self.optimizer_A = torch.optim.Adam(self.unet.parameters(), lr=lr)
                # 紀錄 loss
                log_train.append(loss_train / len(DataLoader_train))
                log_val.append(loss_val / len(DataLoader_val))
                # log_mse.append(mse_val / len(DataLoader_val))
                plt.plot(log_train, label="train")
                plt.plot(log_val, label="val")
                plt.legend()
                plt.xlabel("epoch")
                half = log_train[int(len(log_train) / 2):] + log_val[int(len(log_val) / 2):]
                plt.ylim(min(half), 2 * max(half) - min(half))
                plt.ylabel("loss")
                plt.grid(linestyle="--", alpha=0.3)
                plt.savefig(Path_Pix2pix + "/unet_loss.png")
                plt.close()

                if count > 10:
                    print("break epoch = ", epoch)
                    break

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor, train=True):
        # medianBlur -----------------------------------------------
        # img_B = real_B.numpy()
        # hid = img_B.copy()
        # for i in range(hid.shape[0]):
        #     hid[i, 0, :, :] = cv2.medianBlur(img_B[i, 0, :, :], 5)
        # hid = torch.from_numpy(hid).to(device)
        # medianBlur -----------------------------------------------

        # autoencoder -----------------------------------------------
        # _, hid = self.autoencoder(real_A)
        # hid = hid.detach()
        # autoencoder -----------------------------------------------

        # site -----------------------------------------------
        # img_B = real_B.numpy()
        # hid = img_B.copy()
        # site = cv2.resize(self.site, (192, 256))
        # for i in range(hid.shape[0]):
        #     hid[i, 0, :, :] = img_B[i, 0, :, :] * site
        # hid = torch.from_numpy(hid).to(device)
        # site -----------------------------------------------

        # real_B -----------------------------------------------
        hid = real_B
        # real_B -----------------------------------------------

        real_B = real_B.to(device)
        Tensor = torch.cuda.FloatTensor
        patch = (1, self.opt.model_h // 2 ** 4, self.opt.model_w // 2 ** 4)
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)  # torch.Size([2, 1, 16, 12])
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)  # torch.Size([2, 1, 4, 4])
        #  Train Generators
        if train:
            self.optimizer_G.zero_grad()  # zero the gradient buffers
        # GAN loss
        fake_B = self.generator(real_A)

        pred_fake = self.discriminator(hid, fake_B)
        loss_GAN = self.L1loss(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = self.customLoss(fake_B, real_B)
        weight = 100
        loss_G = loss_GAN + weight * loss_pixel
        if train:
            loss_G.backward()
            self.optimizer_G.step()
        # print(self.generator.module.down1.model[0].weight)

        #  Train Discriminator
        if train and self.train_D:
            self.optimizer_D.zero_grad()
        # Real loss
        pred_real = self.discriminator(hid, real_B)
        loss_real = self.L1loss(pred_real, valid)
        # Fake loss
        pred_fake = self.discriminator(hid, fake_B.detach())
        loss_fake = self.L1loss(pred_fake, fake)
        # Total loss
        loss_D = (loss_fake + loss_real) / 2
        # loss_r_f = (pred_real.mean() + pred_fake.mean()) / 2  # 正表示鑑別器傾向低估 => 猜假
        pred_real = pred_real.mean()
        pred_fake = pred_fake.mean()
        if train and self.train_D:
            loss_D.backward()
            self.optimizer_D.step()

        return loss_D.item(), pred_real.item(), pred_fake.item(), loss_pixel.item(), loss_GAN.item(), fake_B

    def pltconfig(self):
        self.cmap = mpl.colors.ListedColormap(['lightcyan', 'cyan', 'cornflowerblue', 'blue',
                                               'lime', 'limegreen', 'green', 'yellow', 'orange',
                                               'red', 'tab:red', 'brown', 'fuchsia', 'blueviolet'])
        self.cmap.set_under('white')
        self.cmap.set_over('lavender')
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)

    def plt_shp(self, img, filename, title):
        if self.cmap is None:
            self.pltconfig()
        if not os.path.exists(Path_shp):
            print("please upload the shp file at " + Path_shp)
            return
        TWN_CITY = gpd.read_file(Path_shp)
        long, long2, lat, lat2 = pix2pix_data.Get_LongLat(Region)
        x = np.arange(long, long2 + 0.01, 0.0125)
        y = np.arange(lat, lat2 + 0.01, 0.0125)
        x, y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        bounds = self.bounds
        norm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)
        plt.contourf(x, y, img[::-1, :], bounds, cmap=self.cmap)
        ax = TWN_CITY.geometry.plot(ax=ax, alpha=0.3)
        plt.xlim(long, long2)
        plt.ylim(lat, lat2)
        plt.title(title)
        plt.xticks(np.linspace(long, long2, 5))
        if Region == "RS_TW":
            plt.xticks(np.arange(120, 122.5, 0.5))
        plt.colorbar(
            mpl.cm.ScalarMappable(cmap=self.cmap, norm=norm),
            # cax=ax,
            boundaries=[-10] + bounds + [100],
            extend='both',
            extendfrac='auto',
            # ticks=bounds,
            spacing='uniform',
        )
        plt.savefig(filename)
        plt.close()

    def writefile(self, p, filepath):
        return
        with open(filepath, "w") as f:
            c = 0
            for i in range(275):
                for j in range(162):
                    c += 1
                    long = 120 + 0.0125 * i
                    lat = 21.8875 + 0.0125 * j
                    value = ("%.2f" % p[i][-1 - j]).rjust(9, " ")
                    f.writelines("%06s  %3.4f   %2.4f" % (c, long, lat) + value + "\n")

    def cal_site(self, gt, p):
        threshold = self.threshold
        site = self.site
        gt, p = gt[site == 1], p[site == 1]
        self.roc_gt += list(gt)
        self.roc_p += list(p)

        mer = 100 * (np.abs(gt - p) / (gt + 1.01)).mean()
        A = np.logical_and(gt > threshold, p > threshold).sum()
        B = np.logical_and(gt > threshold, p <= threshold).sum()  # 漏報
        C = np.logical_and(gt <= threshold, p > threshold).sum()  # 誤報
        csi = A / (A + B + C) if A + B + C != 0 else None
        mse = np.square(np.subtract(gt, p)).mean()
        self.mbe += list(np.subtract(p, gt))
        self.highmse += list(np.square(np.subtract(gt[gt > 5], p[gt > 5])))
        self.heavymse += list(np.square(np.subtract(gt[gt > 40], p[gt > 40])))
        str = "gt, avg=%.5f, max=%.3f, min=%.3f \np , avg=%.5f, max=%.3f, min=%.3f \n" \
              % (np.mean(gt), np.max(gt), np.min(gt), np.mean(p), np.max(p), np.min(p))
        return mer, csi, str, mse, A, B, C

    def cal(self, gt, p):
        if self.site is not None:
            return self.cal_site(gt, p)

        threshold = self.threshold
        mer = 100 * (np.abs(gt - p) / (gt + 1.01)).mean()
        A = np.logical_and(gt > threshold, p > threshold).sum()
        B = np.logical_and(gt > threshold, p <= threshold).sum()
        C = np.logical_and(gt <= threshold, p > threshold).sum()
        csi = A / (A + B + C) if A + B + C != 0 else None
        mse = np.square(np.subtract(gt, p)).mean()
        str = "gt, avg=%.5f, max=%.3f, min=%.3f \np , avg=%.5f, max=%.3f, min=%.3f \n" \
              % (np.mean(gt), np.max(gt), np.min(gt), np.mean(p), np.max(p), np.min(p))
        return mer, csi, str, mse, A, B, C

    def test(self, DataLoader_test):
        print("pix2pixModel test")
        self.load_model(True)
        doc_test = Path_Pix2pix + "/test" + datetime.datetime.today().strftime("_%m%d_%H%M")
        if not os.path.exists(doc_test):
            os.mkdir(doc_test)
        self.pltconfig()
        if self.site is not None:
            print("test by site.npy")
        with open(doc_test + "/cal.txt", "w") as f:
            f.writelines(Path_Pix2pix + "\n")
        temp_mer, temp_mse, temp_csi, temp_avg, temp_max = [], [], [], [], []
        self.roc_p, self.roc_gt, self.highmse, self.heavymse, self.mbe = [], [], [], [], []
        A, B, C = 0, 0, 0

        self.generator.eval()
        with torch.no_grad():
            loss_test = 0
            for _, (test_x, test_y, X_time, YName) in enumerate(DataLoader_test):
                output = self.generator(test_x).squeeze(1)
                loss_test += self.MSELoss(output, test_y.float().to(device)).item()
                for b in range(test_x.shape[0]):
                    p = output[b, :, :].cpu().numpy().reshape(self.opt.model_h, self.opt.model_w)
                    if log2_pre:
                        p = 2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1
                    else:
                        p *= pix2pix_data.MaxPre
                    if Region == "RS_TW":
                        p = cv2.resize(p, (self.opt.img_w, self.opt.img_h))

                    # gt = np.load(YName[b])
                    # gt = pix2pix_data.SplitRegion(gt, "TW", Region)
                    doc = doc_test + "/" + YName[b][-40:-4]  # qpepre_202105010600-202105010700_1_h
                    # 存在 label
                    if os.path.exists(YName[b]):
                        gt = np.load(YName[b])
                        gt = pix2pix_data.SplitRegion(gt, "TW", Region)
                        mer, csi, log_str, mse, cal_a, cal_b, cal_c = self.cal(gt, p)
                        if mse < 30 and np.max(gt) > 80:
                            if not os.path.exists(doc):
                                os.mkdir(doc)
                            self.plt_shp(p, doc + "/p" + ".png", YName[b][-33:-8])
                            self.writefile(p, doc + "/QPE_" + YName[b][-40:-4] + ".txt")
                            if self.opt.pltSource:
                                for t in X_time:
                                    self.pltRadar(t[0], doc)
                            self.plt_shp(gt, doc + "/gt" + ".png", YName[b][-33:-8])
                            with open(doc + "/cal.txt", "w") as f:
                                f.writelines("MER=%.5f, MSE=%.3f, " % (mer, mse)
                                             + ("CSI=NaN\n" if csi is None else ("CSI=%.3f\n" % csi)))
                                f.writelines(log_str)
                        if mse > 30:
                            doc = doc + " MSE = %.2f" % mse
                            if not os.path.exists(doc):
                                os.mkdir(doc)
                            self.plt_shp(p, doc + "/p" + ".png", YName[b][-33:-8])
                            self.writefile(p, doc + "/QPE_" + YName[b][-40:-4] + ".txt")
                            if self.opt.pltSource:
                                for t in X_time:
                                    self.pltRadar(t[0], doc)
                            self.plt_shp(gt, doc + "/gt" + ".png", YName[b][-33:-8])
                            with open(doc + "/cal.txt", "w") as f:
                                f.writelines("MER=%.5f, MSE=%.3f, " % (mer, mse)
                                             + ("CSI=NaN\n" if csi is None else ("CSI=%.3f\n" % csi)))
                                f.writelines(log_str)

                        max_gt, mean_gt = np.max(gt), np.mean(gt)
                        temp_mer.append(mer)
                        temp_mse.append(mse)
                        temp_csi.append(csi)
                        temp_avg.append(mean_gt)
                        temp_max.append(max_gt)
                        A += cal_a
                        B += cal_b
                        C += cal_c

                        with open(doc_test + "/cal.txt", "a") as f:
                            cal_log = "%s : MER=%.5f, MSE=%.3f, " % (YName[b][-40:-4], mer, mse) \
                                      + ("CSI=NaN" if csi is None else ("CSI=%.3f" % csi))
                            # print(cal_log)
                            f.writelines(cal_log + "\n")
                            f.writelines(log_str)
        if len(temp_mer) == 0:
            return
        loss_test *= 1000

        # print("DataLoader_test: ", loss_test / len(DataLoader_test))

        def avg_(list, label):
            if len(list) == 0:
                return label + "=NaN"
            return label + "=%.3f" % (sum(list) / len(list))

        with open(doc_test + "/cal.txt", "a") as f:
            CSI = A / (A + B + C) if A + B + C != 0 else 0
            POD = A / (A + B) if A + B != 0 else 0
            FAR = C / (A + C) if A + C != 0 else 0
            f.writelines("Sum : MER=%.5f, CSI=%.3f, POD=%.3f, FAR=%.3f, MSE=%.3f, hMSE=%.3f, HMSE=%.3f , MBE=%.3f  \n"
                         % (sum(temp_mer) / len(temp_mer), CSI, POD, FAR, sum(temp_mse) / len(temp_mse),
                            sum(self.highmse) / len(self.highmse), sum(self.heavymse) / len(self.heavymse)
                            , sum(self.mbe) / len(self.mbe)))
            # f.writelines("topcsi_avg=%.3f, topcsi_max=%.3f \n" % (sum_csi_and_avg / sum_csi_or_avg,
            #                                                       sum_csi_and_max / sum_csi_or_max))
            topmer_avg = list(
                filter(lambda x: list(map(lambda te: float(te) > 1, temp_avg))[temp_mer.index(x)] is True, temp_mer))
            topmer_max = list(
                filter(lambda x: list(map(lambda te: int(te) >= 40, temp_max))[temp_mer.index(x)] is True, temp_mer))
            f.writelines(avg_(topmer_avg, "topmer_avg") + ", " + avg_(topmer_max, "topmer_max") + " \n")
            topmse_avg = list(
                filter(lambda x: list(map(lambda te: float(te) > 1, temp_avg))[temp_mse.index(x)] is True, temp_mse))
            topmse_max = list(
                filter(lambda x: list(map(lambda te: int(te) >= 40, temp_max))[temp_mse.index(x)] is True, temp_mse))
            f.writelines(avg_(topmse_avg, "topmse_avg") + ", " + avg_(topmse_max, "topmse_max") + " \n")

        # print("Mean error rate = %.5f " % (sum(temp_mer) / len(temp_mer)) + "%")
        print("Critical Success Index = %.5f " % CSI)
        print("Probability of Detection = %.5f " % POD)
        print("False Alarm Ratio = %.5f " % FAR)
        print("Mean Square Error = %.5f " % (sum(temp_mse) / len(temp_mse)))
        print("High Mean Square Error = %.5f " % (sum(self.highmse) / len(self.highmse)))
        print("Heavy Mean Square Error = %.5f " % (sum(self.heavymse) / len(self.heavymse)))
        print("Mean Bias Error = %.5f " % (sum(self.mbe) / len(self.mbe)))

        # 繪圖
        # plt.hist(temp_mer)
        # plt.xlabel("Mean error rate (%)")
        # plt.ylabel("Count")
        # plt.savefig(doc_test + "/mer.png")
        # plt.close()

        temp_csi_noNone = [csi for csi in temp_csi if csi is not None]
        plt.hist(temp_csi_noNone)
        plt.xlabel("Critical Success Index")
        plt.ylabel("Count")
        plt.savefig(doc_test + "/csi.png")
        plt.close()

        # avg_csi = list(
        #     filter(lambda x: list(map(lambda csi: csi is not None, temp_csi))[temp_avg.index(x)] is True, temp_avg))
        # plt.scatter(avg_csi, temp_csi_noNone, s=30, c='blue', marker='s',
        #             alpha=0.9, linewidths=0.3, edgecolors='red')
        # plt.xlabel('Average rain fall (mm/hr)')
        # plt.ylabel('Critical Success Index')
        # plt.savefig(doc_test + "/avg_csi.png")
        # plt.close()

        # max_csi = list(
        #     filter(lambda x: list(map(lambda csi: csi is not None, temp_csi))[temp_max.index(x)] is True, temp_max))
        # plt.scatter(max_csi, temp_csi_noNone, s=30, c='blue', marker='s',
        #             alpha=0.9, linewidths=0.3, edgecolors='red')
        # plt.xlabel('Maximum rain fall (mm/hr)')
        # plt.ylabel('Critical Success Index')
        # plt.savefig(doc_test + "/max_csi.png")
        # plt.close()

        # plt.scatter(temp_avg, temp_mer, s=30, c='blue', marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
        # plt.xlabel('Average rain fall (mm/hr)')
        # plt.ylabel('Mean error rate (%)')
        # plt.grid(linestyle="--", alpha=0.3)
        # plt.savefig(doc_test + "/avg_mer.png")
        # plt.close()

        # plt.scatter(temp_max, temp_mer, s=30, c='blue', marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
        # plt.xlabel('Maximum rain fall (mm/hr)')
        # plt.ylabel('Mean error rate (%)')
        # plt.grid(linestyle="--", alpha=0.3)
        # plt.savefig(doc_test + "/max_mer.png")
        # plt.close()

        plt.scatter(temp_avg, temp_mse, s=30, c='blue', marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
        plt.xlabel('Average rain fall (mm/hr)')
        plt.ylabel('Mean square error')
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(doc_test + "/avg_mse.png")
        plt.close()

        plt.scatter(temp_max, temp_mse, s=30, c='blue', marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
        plt.xlabel('Maximum rain fall (mm/hr)')
        plt.ylabel('Mean square error')
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(doc_test + "/max_mse.png")
        plt.close()

        x = np.array(self.roc_gt).reshape(-1, 1)
        y = np.array(self.roc_p).reshape(-1, 1)
        # 一元線性迴歸：最小二乘法(OLS)
        modelRegL = LinearRegression()  # 建立線性迴歸模型
        modelRegL.fit(x, y)  # 模型訓練：資料擬合
        yFit = modelRegL.predict(x)  # 用迴歸模型來預測輸出

        # 輸出迴歸結果 XUPT
        print('迴歸截距: w0={}'.format(modelRegL.intercept_))  # w0: 截距
        print('迴歸係數: w1={}'.format(modelRegL.coef_))  # w1,..wm: 迴歸係數

        # 迴歸模型的評價指標 YouCans
        print('R2 確定係數：{:.4f}'.format(modelRegL.score(x, y)))  # R2 判定係數
        print('均方誤差：{:.4f}'.format(mean_squared_error(y, yFit)))  # MSE 均方誤差
        print('平均絕對值誤差：{:.4f}'.format(mean_absolute_error(y, yFit)))  # MAE 平均絕對誤差
        print('中位絕對值誤差：{:.4f}'.format(median_absolute_error(y, yFit)))  # 中值絕對誤差

        plt.scatter(x, y, s=10, c='r', marker='.', alpha=0.9)
        plt.xlabel('ground truth (mm/hr)')
        plt.ylabel('predict result (mm/hr)')
        plt.plot([0, 100], [0, 100], color='k', lw=1, linestyle='-')
        plt.plot(x, yFit, lw=1, linestyle='--')
        plt.xlim([0.0, 100])
        plt.ylim([0.0, 100])
        plt.grid(linestyle="--", alpha=0.3)
        tit = "y = %.3fx + %.3f, r2 = %.3f" % (modelRegL.coef_[0][0], modelRegL.intercept_[0], modelRegL.score(x, y))
        plt.title(tit)
        plt.savefig(doc_test + "/gt_p.png")
        plt.close()

        if len(self.roc_p) != 0:
            try:
                print(len(self.roc_gt), len(self.roc_p))
                fpr, tpr, threshold = roc_curve(self.roc_gt, self.roc_p)  # 計算真正率和假正率
                roc_auc = auc(fpr, tpr)  # 計算auc的值
                plt.figure()
                lw = 2
                # plt.figure(figsize=(10, 10))
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率為橫座標，真正率為縱座標做曲線
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.01])
                plt.ylim([0.0, 1.01])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
                plt.legend(loc="lower right")
                plt.savefig(doc_test + "/roc.png")
                plt.close()
            except Exception as e:
                print(pix2pix_data.errMsg(e))

    def testUnet(self, DataLoader_test):
        print("testUnet test")
        # 設定
        self.generator_Load_Unet = True
        # test
        self.test(DataLoader_test)
        # 還原
        self.generator = None
        self.generator_Load_Unet = False

    def pltRadar(self, file, doc_pltRadar=""):
        if not os.path.exists(Path_shp):
            print("please upload the shp file at ./shp/TWN_CITY.shp")
            return

        try:
            time = file[-16:-4]
            type = ""
            for t in ["DR", "DZ", "KD", "WD"]:
                if t in file:
                    type = t

            dic = {"DR": "ZDR", "DZ": "ZH", "KD": "KDP", "WD": "WISSDOM"}

            if type == "WD":
                (img_u, img_v, img_w) = pix2pix_data.SourceToNpy(pix2pix_data.TimeToSourcePath(time, type), type)
                cmap_wd = mpl.colors.ListedColormap(['blue', 'cornflowerblue', 'cyan', 'lightcyan', 'white',
                                                     'yellow', 'orange', 'red', 'tab:red', ])
                level_bound_WD = [-2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5]
            else:
                img = pix2pix_data.SourceToNpy(pix2pix_data.TimeToSourcePath(time, type), type)
                if img is None:
                    print(time, type, "is None")
                    return
                cmap = mpl.colors.ListedColormap(['white', 'lightcyan', 'cyan', 'cornflowerblue', 'blue',
                                                  'lime', 'limegreen', 'green', 'yellow', 'orange',
                                                  'red', 'tab:red', 'brown', 'fuchsia', 'blueviolet'])
                levels = 15
                level_bound = np.linspace(np.min(img), np.max(img), levels)
                if type == "KD":
                    level_bound = np.linspace(np.min(img), np.max(img) / 10, levels)

            TWN_CITY = gpd.read_file(Path_shp)
            plt.figure(figsize=(12, 10))
            for i in range(1, 9):
                plt.subplot(2, 4, i)
                if type == "WD":
                    x = np.arange(119, 122.801, 0.01)
                    y = np.arange(21, 26.00001, 0.01)
                    x, y = np.meshgrid(x, y)
                    plt.contourf(x, y, img_w[::-1, :, i - 1], level_bound_WD, cmap=cmap_wd)
                    x = np.arange(119, 122.801, .3)
                    y = np.arange(21, 26.00001, .3)
                    x, y = np.meshgrid(x, y)
                    u = img_u[::-1, :, i - 1][::30, ::30]
                    v = img_v[::-1, :, i - 1][::30, ::30]
                    ax = plt.gca()
                    q = ax.quiver(x, y, u, v)
                    TWN_CITY.geometry.plot(ax=ax, alpha=0.3)
                    plt.xlim(119, 122.8)
                    plt.ylim(21, 26)
                    plt.xticks(np.arange(120, 122.8, 2))
                    if i in [1, 5]:
                        plt.yticks(np.arange(22, 26, 2))
                    else:
                        plt.yticks([])
                else:
                    x = np.arange(118, 123.51, 0.0125)
                    y = np.arange(20, 27.00001, 0.0125)
                    x, y = np.meshgrid(x, y)
                    plt.contourf(x, y, img[::-1, :, i - 1], level_bound, cmap=cmap)
                    ax = plt.gca()
                    TWN_CITY.geometry.plot(ax=ax, alpha=0.3)
                    plt.xlim(118, 123.5)
                    plt.ylim(20, 27)
                    plt.xticks(np.arange(119, 123.5, 2))
                    if i in [1, 5]:
                        plt.yticks(np.arange(21, 27, 2))
                    else:
                        plt.yticks([])
                plt.title(str(i * 0.5) + "km ")
                plt.colorbar()
            plt.suptitle(dic[type] + " " + time)
            plt.savefig(doc_pltRadar + "/" + type + ".png")
            plt.close()
        except Exception as e:
            print(pix2pix_data.errMsg(e))
            return

    def save_model(self):
        if not os.path.exists(Path_Pix2pix):
            os.mkdir(Path_Pix2pix)
        for trial in range(2):
            try:
                torch.save(self.generator.module.state_dict(), Path_Pix2pix + "/generator.pth")
                torch.save(self.discriminator.module.state_dict(), Path_Pix2pix + "/discriminator.pth")
                with open(Path_Pix2pix + "/config.txt", "w") as f:
                    f.writelines(str(self.opt))
                break
            except:
                print('saving failed')

    def save_unet(self):
        if not os.path.exists(Path_Pix2pix):
            os.mkdir(Path_Pix2pix)
        for trial in range(2):
            try:
                torch.save(self.unet.module.state_dict(), Path_Pix2pix + "/unet.pth")
                with open(Path_Pix2pix + "/config.txt", "w") as f:
                    f.writelines(str(self.opt))
                break
            except:
                print('saving failed')

    def load_model(self, exist, model_path=None):
        self.discriminator = nn.DataParallel(Discriminator().to(device))

        if self.generator_Load_Unet:  # 將unet.pth load進self.generator, for test使用
            self.load_unet(exist)
            self.generator = self.unet
            return True
        if self.Unet_replace_generator:  # 將UNet load進self.generator, for train使用
            self.generator = nn.DataParallel(UNet(24 + (48 if useWD else 0), 1).to(device))
        else:
            self.generator = nn.DataParallel(GeneratorUNet().to(device))
        load_path = Path_Pix2pix if model_path is None else model_path + r"/QPE/pix2pix"
        if exist:
            if os.path.exists(load_path):
                print("load model at " + load_path + "/generator.pth")
                self.generator.module.load_state_dict(torch.load(load_path + "/generator.pth"))
                print("load model at " + load_path + "/discriminator.pth")
                self.discriminator.module.load_state_dict(torch.load(load_path + "/discriminator.pth"))
                return True
            print("cannot find model at " + load_path + "/generator.pth")
        return False

    def load_autoencoder(self, exist, model_path=None):
        self.autoencoder = nn.DataParallel(AutoEncoder(24 + (48 if useWD else 0)).to(device))
        load_path = Path_Pix2pix if model_path is None else model_path + r"/QPE/pix2pix"
        if exist and os.path.exists(load_path):
            print("load model at " + load_path + "/autoencoder.pth")
            self.autoencoder.module.load_state_dict(torch.load(load_path + "/autoencoder.pth"))
            return True
        return False

    def load_unet(self, exist, model_path=None):
        self.unet = nn.DataParallel(UNet(24 + (48 if useWD else 0), 1).to(device))
        load_path = Path_Pix2pix if model_path is None else model_path + r"/QPE/pix2pix"
        if exist and os.path.exists(load_path):
            print("load model at " + load_path + "/unet.pth")
            self.unet.module.load_state_dict(torch.load(load_path + "/unet.pth"))
            return True
        return False

    def predict_point(self, long=120.2420, lat=24.5976, start="2021/05/28 00:00", end="2021/05/31 19:00"):
        X, Y, timelist = CreateContinuousData(start, end)
        predict_path = os.path.join(Path_Pix2pix, "predict")
        if not os.path.exists(predict_path):
            os.mkdir(predict_path)

        TWN_CITY = gpd.read_file(pix2pix_data.Path_shp)
        ax = plt.gca()
        plt.scatter(long, lat, s=20, c='r', marker="*")
        TWN_CITY.geometry.plot(ax=ax, alpha=0.3)
        x1, x2, y1, y2 = pix2pix_data.Get_LongLat(Region)
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
        plt.savefig(os.path.join(predict_path, "%.3f°E_%.3f°N.png" % (long, lat)))
        plt.close()

        gt_point = []
        a, _, c, _ = pix2pix_data.Get_LongLat("TW")
        i, j = round((c - lat) / 0.0125) - 1, round((long - a) / 0.0125)  # 座標點 ex:-50, 56
        for y in Y:
            gt = np.load(y[0])
            gt_point.append(gt[i][j])

        def oldGet_predict(model_path, lat, long, region, X, Y):
            # region : NT, ST, RS_TW
            point = []
            if self.load_model(True, model_path):
                a, _, c, _ = pix2pix_data.Get_LongLat(region)
                model_h, model_w = pix2pix_data.Get_LongLat(region, size=True)
                img_h, img_w = pix2pix_data.Get_LongLat(region, size=True)
                if region == "RS_TW":
                    img_h, img_w = pix2pix_data.Get_LongLat("TW", size=True)
                i, j = round((c - lat) / 0.0125) - 1, round((long - a) / 0.0125)  # 座標點 ex:-50, 56
                if -i >= img_h or j >= img_w:
                    print(i, j, img_h, img_w, "指定的座標點超出預測範圍")
                    return
                print(X[2][2])
                print("predict data: ", end="")
                predict_data = DataLoader(NNDataset(X, Y, True), batch_size=1, num_workers=1)
                with torch.no_grad():
                    for _, (test_x, _, _, YName) in enumerate(predict_data):
                        output = self.generator(test_x).squeeze(1)
                        for b in range(test_x.shape[0]):
                            p = output[b, :, :].cpu().numpy().reshape(model_h, model_w)
                            # print(np.max(p), end=" ")
                            if log2_pre:
                                p = 2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1
                            else:
                                p *= pix2pix_data.MaxPre
                            if region == "RS_TW":
                                p = cv2.resize(p, (img_w, img_h))
                            point.append(p[i][j])
                            # print(p[i][j])
            return point

        def Get_predict(lat, long, region, X, Y):
            # region : NT, ST, RS_TW
            point = []
            a, _, c, _ = pix2pix_data.Get_LongLat(region)
            model_h, model_w = pix2pix_data.Get_LongLat(region, size=True)
            img_h, img_w = pix2pix_data.Get_LongLat(region, size=True)
            if region == "RS_TW":
                img_h, img_w = pix2pix_data.Get_LongLat("TW", size=True)
            i, j = round((c - lat) / 0.0125) - 1, round((long - a) / 0.0125)  # 座標點 ex:-50, 56
            if -i >= img_h or j >= img_w:
                print(i, j, img_h, img_w, "指定的座標點超出預測範圍")
                return
            print(X[2][2])
            print("predict data: ", end="")
            predict_data = DataLoader(NNDataset(X, Y, True), batch_size=1, num_workers=1)
            with torch.no_grad():
                for _, (test_x, _, _, YName) in enumerate(predict_data):
                    output = self.generator(test_x).squeeze(1)
                    for b in range(test_x.shape[0]):
                        p = output[b, :, :].cpu().numpy().reshape(model_h, model_w)
                        # print(np.max(p), end=" ")
                        if log2_pre:
                            p = 2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1
                        else:
                            p *= pix2pix_data.MaxPre
                        if region == "RS_TW":
                            p = cv2.resize(p, (img_w, img_h))
                        point.append(p[i][j])
                        # print(p[i][j])
            return point

        # 全台部分
        # if self.load_model(True, r"E:/TWCC"):
        #     a, b, c, d = pix2pix_data.Get_LongLat("RS_TW")
        #     i, j = round((c - lat) / 0.0125) - 1, round((long - a) / 0.0125)  # 座標點
        #     X_tw = [[path.replace("/" + Region + "/", "/TWCC/") for path in data] for data in X]
        #     print(X_tw[2][2])
        #     print("predict data: ", end="")
        #     predict_data = DataLoader(NNDataset(X_tw, Y, True), batch_size=1, num_workers=1)
        #     with torch.no_grad():
        #         for _, (test_x, _, _, _) in enumerate(predict_data):
        #             output = self.generator(test_x).squeeze(1)
        #             for b in range(test_x.shape[0]):
        #                 p2 = output[b, :, :].cpu().numpy().reshape(256, 192)
        #                 print(np.max(p2), end=" ")
        #                 p2 = 2 ** (p2 * np.log2(pix2pix_data.MaxPre + 1)) - 1
        #                 p2 = cv2.resize(2 ** (p2 * np.log2(pix2pix_data.MaxPre + 1)) - 1, (162, 275))
        #                 p_tw_point.append(p2[i][j])
        #                 print(p2[i][j])

        # self.generator = nn.DataParallel(UNet(24 + (48 if useWD else 0), 1).to(device))
        self.generator = nn.DataParallel(GeneratorUNet().to(device))
        self.generator.module.load_state_dict(torch.load(r"E:\TWCC1\QPE\pix2pix\test_0312_0332\generator.pth"))
        p1_point = Get_predict(lat, long, Region, X, Y)
        self.generator = nn.DataParallel(UNet(24 + (48 if useWD else 0), 1).to(device))
        self.generator.module.load_state_dict(torch.load(r"E:\TWCC1\QPE\pix2pix\test_0206_0219\unet.pth"))
        p2_point = Get_predict(lat, long, Region, X, Y)
        # X_tw = [[path.replace("/" + Region + "/", "/TWCC/") for path in data] for data in X]
        # p_tw_point = Get_predict(r"E:/TWCC", lat, long, "RS_TW", X_tw, Y)

        with open(os.path.join(predict_path, "predict_%.3f°E_%.3f°N.txt" % (long, lat)), "w") as f:
            f.writelines("Time              Lat        Long     predict    Observation \n")
            for i in range(len(p1_point)):
                pvalue = ("%.2f" % p1_point[i]).rjust(9, " ")
                value = ("%.2f" % gt_point[i]).rjust(9, " ")
                f.writelines("%s  %3.4f   %2.4f" % (datetime.datetime.strftime(timelist[i], "%Y/%m/%d %H:%M"),
                                                    long, lat) + pvalue + value + "\n")

        if len(timelist) < 24 * 4:
            sample_tick = [t for t in timelist if t.hour in [0, 6, 12, 18]]
            tick = [datetime.datetime.strftime(t, "%Y/%m/%d %H:%M") for t in sample_tick]
            tick_label = [datetime.datetime.strftime(t, "%m/%d") if t.hour == 0 else str(t.hour) for t in sample_tick]
        else:
            sample_tick = timelist[::int(len(timelist) / 6)]
            tick = [t + " 00:00" for t in [datetime.datetime.strftime(t, "%Y/%m/%d") for t in sample_tick]]
            tick_label = [datetime.datetime.strftime(t, "%m/%d") for t in sample_tick]
        plotlist = [datetime.datetime.strftime(t, "%Y/%m/%d %H:%M") for t in timelist]
        plt.bar(plotlist, gt_point, color='cornflowerblue', label="Observation", alpha=0.75)
        plt.plot(plotlist, p1_point, color='r', marker='.', label="ours", fillstyle='none')
        plt.plot(plotlist, p2_point, color='g', marker='.', label="Unet", fillstyle='none')
        print("ours: ", p1_point)
        print("Unet: ", p2_point)
        plt.xticks(tick, tick_label)
        plt.xlabel(start + " ~ " + end)
        plt.ylabel("mm/hr")
        plt.ylim(bottom=0)
        plt.legend()
        plt.title("%.3f°E %.3f°N" % (long, lat))
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(os.path.join(predict_path, "predict_%.3f°E_%.3f°N.png" % (long, lat)))


def CreateDataSet(opt):
    # qpepre_202005010000-202005010100_1_h
    # CAPPI_COMP_DZ_202005010000 DR KD
    # wissdom_out_Taiwan_mosaic_202105020030
    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/DZ", Path_QPE + r"/KD", Path_QPE + r"/WD"
    X_train, X_test, Y_train, Y_test, X_val, Y_val = [], [], [], [], [], []
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), os.listdir(
        KD), os.listdir(WD)
    if len(DZList) == 0:
        DZ = Path_QPE + r"/Z"
        DZList = os.listdir(Path_QPE + r"/Z")

    for pre in preList:
        b, e = pre[7:19], pre[20:32]
        b_datatime = datetime.datetime.strptime(b, "%Y%m%d%H%M")
        # if b_datatime.year != 2021:
        #     continue
        TestData = b_datatime.month == 6 and b_datatime.year == 2021
        if TestData and b[-2:] != "00":
            continue  # 測試資料只拿整點
        h = datetime.datetime.strftime(datetime.datetime.strptime(b, "%Y%m%d%H%M")
                                       + datetime.timedelta(minutes=30), "%Y%m%d%H%M")
        dr, dz, kd, wd = pix2pix_data.BindName(h, ".npy")
        if dr in DRList and dz in DZList and kd in KDList:
            if useWD and wd not in WDList:
                continue
            # if wd not in WDList:
            #     continue
            Augmentation = False
            if TestData:  # 測試資料
                if useWD:
                    X_test.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                else:
                    X_test.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                Y_test.append([qpepre + r"/" + pre, 0])
            else:  # 訓練資料
                avg = np.mean(np.load(qpepre + r"/" + pre))
                max = np.max(np.load(qpepre + r"/" + pre))
                if max < opt.Threshold and avg < opt.Threshold_Avg:
                    continue
                if opt.DataAugmentation and max > opt.AugmentationThreshold:
                    Augmentation = True
                for i in range(4) if Augmentation else range(1):
                    if useWD:
                        X_train.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                    else:
                        X_train.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                    Y_train.append([qpepre + r"/" + pre, i])

    # print("X_train = ", len(X_train), ", X_train_wd = ", len(X_train_wd))
    # print("X_test = ", len(X_test), ", X_test_wd = ", len(X_test_wd))
    # spilt
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=opt.train_size, random_state=0)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def CreateDataSet2(opt, WDData=True):
    # qpepre_202005010000-202005010100_1_h
    # CAPPI_COMP_DZ_202005010000 DR KD
    # wissdom_out_Taiwan_mosaic_202105020030
    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/Z", Path_QPE + r"/KD", Path_QPE + r"/WD"
    X_train, X_test, Y_train, Y_test = [], [], [], []
    X_train_wd, X_test_wd, Y_train_wd, Y_test_wd = [], [], [], []
    X_val, X_val_wd, Y_val, Y_val_wd = [], [], [], []
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), os.listdir(
        KD), os.listdir(WD)

    TestList = [i.strip() for i in opt.TestData.split(",")]
    for pre in preList:
        b = pre[7:19]
        if b[-2:] != "00":
            continue
        # hourlist = [b[:-2] + "00", b[:-2] + "10", b[:-2] + "20", b[:-2] + "30", b[:-2] + "40", b[:-2] + "50"]
        hourlist = [b[:-2] + "30"]
        TestData = False
        for t in TestList:
            if b[:len(t)] == t:
                TestData = True
                hourlist = [b[:-2] + "30"]
        for h in hourlist:
            dr, dz, kd, wd = pix2pix_data.BindName(h, ".npy")
            if dr in DRList and dz in DZList and kd in KDList:
                Augmentation = False
                HaveWD = False
                if TestData:
                    if wd in WDList and WDData:
                        X_test_wd.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                        Y_test_wd.append([qpepre + r"/" + pre, 0])
                    else:
                        X_test.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                        Y_test.append([qpepre + r"/" + pre, 0])
                else:
                    avg = np.mean(np.load(qpepre + r"/" + pre))
                    max = np.max(np.load(qpepre + r"/" + pre))
                    if max < opt.Threshold and avg < opt.Threshold_Avg:
                        continue
                    if opt.DataAugmentation and max > opt.AugmentationThreshold:
                        Augmentation = True
                    if wd in WDList:
                        HaveWD = True
                    for i in range(4) if Augmentation else range(1):
                        if HaveWD and WDData:
                            X_train_wd.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                            Y_train_wd.append([qpepre + r"/" + pre, i])
                        X_train.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                        Y_train.append([qpepre + r"/" + pre, i])

    # print("X_train = ", len(X_train), ", X_train_wd = ", len(X_train_wd))
    # print("X_test = ", len(X_test), ", X_test_wd = ", len(X_test_wd))
    # spilt
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=opt.train_size, random_state=0)
    if len(X_train_wd) != 0:
        X_train_wd, X_val_wd, Y_train_wd, Y_val_wd = train_test_split(X_train_wd, Y_train_wd, train_size=opt.train_size,
                                                                      random_state=0)
    # merge
    btr, bva, bte = opt.batch_size_train, opt.batch_size_val, opt.batch_size_test
    X_train, Y_train = X_train[:(len(X_train) // btr) * btr] + X_train_wd, \
                       Y_train[:(len(X_train) // btr) * btr] + Y_train_wd
    X_val, Y_val = X_val[:(len(X_val) // bva) * bva] + X_val_wd, Y_val[:(len(X_val) // bva) * bva] + Y_val_wd
    X_test, Y_test = X_test + X_test_wd, Y_test + Y_test_wd

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def oldCreateDataSet(opt, WDData=True):
    # qpepre_202005010000-202005010100_1_h
    # CAPPI_COMP_DZ_202005010000 DR KD
    # wissdom_out_Taiwan_mosaic_202105020030
    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/DZ", Path_QPE + r"/KD", Path_QPE + r"/WD"
    X_train, X_test, Y_train, Y_test = [], [], [], []
    X_train_wd, X_test_wd, Y_train_wd, Y_test_wd = [], [], [], []
    X_val, X_val_wd, Y_val, Y_val_wd = [], [], [], []
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), os.listdir(
        KD), os.listdir(WD)

    TestList = [i.strip() for i in opt.TestData.split(",")]
    for pre in preList:
        b, e = pre[7:19], pre[20:32]
        TestData = False
        for t in TestList:
            if b[:len(t)] == t:
                TestData = True
        if TestData and b[-2:] != "00":
            continue  # 測試資料只拿整點
        h = datetime.datetime.strftime(datetime.datetime.strptime(b, "%Y%m%d%H%M")
                                       + datetime.timedelta(minutes=30), "%Y%m%d%H%M")
        dr, dz, kd, wd = pix2pix_data.BindName(h, ".npy")
        if dr in DRList and dz in DZList and kd in KDList:
            Augmentation = False
            HaveWD = False
            if TestData:
                if wd in WDList and WDData:
                    X_test_wd.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                    Y_test_wd.append([qpepre + r"/" + pre, 0])
                else:
                    X_test.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                    Y_test.append([qpepre + r"/" + pre, 0])
            else:
                avg = np.mean(np.load(qpepre + r"/" + pre))
                max = np.max(np.load(qpepre + r"/" + pre))
                if max < opt.Threshold and avg < opt.Threshold_Avg:
                    continue
                if opt.DataAugmentation and max > opt.AugmentationThreshold:
                    Augmentation = True
                if wd in WDList:
                    HaveWD = True
                for i in range(4) if Augmentation else range(1):
                    if HaveWD and WDData:
                        X_train_wd.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                        Y_train_wd.append([qpepre + r"/" + pre, i])
                    X_train.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                    Y_train.append([qpepre + r"/" + pre, i])

    # print("X_train = ", len(X_train), ", X_train_wd = ", len(X_train_wd))
    # print("X_test = ", len(X_test), ", X_test_wd = ", len(X_test_wd))
    # spilt
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=opt.train_size, random_state=0)
    if len(X_train_wd) != 0:
        X_train_wd, X_val_wd, Y_train_wd, Y_val_wd = train_test_split(X_train_wd, Y_train_wd, train_size=opt.train_size,
                                                                      random_state=0)
    # merge
    btr, bva, bte = opt.batch_size_train, opt.batch_size_val, opt.batch_size_test
    X_train, Y_train = X_train[:(len(X_train) // btr) * btr] + X_train_wd, \
                       Y_train[:(len(X_train) // btr) * btr] + Y_train_wd
    X_val, Y_val = X_val[:(len(X_val) // bva) * bva] + X_val_wd, Y_val[:(len(X_val) // bva) * bva] + Y_val_wd
    X_test, Y_test = X_test + X_test_wd, Y_test + Y_test_wd

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def CreateTestData(opt, WDData=True):
    def NextTime(time, type="h"):
        time = time.ljust(12, '0')
        year = int(time[:4])
        month = int(time[4:6])
        day = int(time[6:8])
        hour = int(time[8:10])
        minute = int(time[10:12])

        if type == "h":
            hour += 1
        if type == "m":
            minute += 10
        if minute == 60:
            minute = 0
            hour += 1
        if hour == 24:
            hour = 0
            day += 1
        if month in [1, 3, 5, 7, 8, 10] and day == 32:
            month += 1
            day = 1
        if month in [4, 6, 9, 11] and day == 31:
            month += 1
            day = 1
        if month == 2 and day == 29:
            month += 1
            day = 1
        if month == 12 and day == 32:
            month = 1
            day = 1

        return "%02d%02d%02d%02d%02d" % (year, month, day, hour, minute)

    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/DZ", Path_QPE + r"/KD", Path_QPE + r"/WD"
    X_test, X_test_wd, Y_test, Y_test_wd = [], [], [], []
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), os.listdir(
        KD), os.listdir(WD)

    TestList = [i.strip() for i in opt.TestData.split(",")]
    for dz in DRList:
        h = dz[14:26]
        TestData = False
        for t in TestList:
            if h[:len(t)] == t and h[-2:] == "30":
                TestData = True
        if TestData:
            dr, dz, kd = "CAPPI_COMP_DR_" + h + ".npy", "CAPPI_COMP_DZ_" + h + ".npy", "CAPPI_COMP_KD_" + h + ".npy"
            wd = "wissdom_out_Taiwan_mosaic_" + h + ".npy"
            if dr in DRList and dz in DZList and kd in KDList:
                HaveWD = False
                if TestData:
                    X, Y = X_test, Y_test
                    if wd in WDList:
                        HaveWD = True
                        X, Y = X_test_wd, Y_test_wd
                for i in range(1):
                    if HaveWD and WDData:
                        X.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
                    else:
                        X.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
                    Y.append([qpepre + r"/qpepre_" + h[:-2] + "00-" + NextTime(h[:-2]) + "_1_h.npy", i])

    # merge
    X_test, Y_test = X_test + X_test_wd, Y_test + Y_test_wd

    return X_test, Y_test


def CreateContinuousData(start="2021/05/01 00:00", end="2021/06/01 00:00", WDData=True):
    timelist = []
    start, end = datetime.datetime.strptime(start, "%Y/%m/%d %H:%M"), \
                 datetime.datetime.strptime(end, "%Y/%m/%d %H:%M")
    while start <= end:
        timelist.append(start)
        start += datetime.timedelta(hours=1)

    X, Y = [], []
    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/DZ", Path_QPE + r"/KD", Path_QPE + r"/WD"
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), \
                                              os.listdir(KD), os.listdir(WD)
    if len(DZList) == 0:
        DZ = Path_QPE + r"/Z"
        DZList = os.listdir(Path_QPE + r"/Z")
    for i in range(len(timelist) - 1):
        now_time = datetime.datetime.strftime(timelist[i], "%Y%m%d%H%M")
        half_time = datetime.datetime.strftime(timelist[i] + datetime.timedelta(minutes=30), "%Y%m%d%H%M")
        next_time = datetime.datetime.strftime(timelist[i + 1], "%Y%m%d%H%M")
        dr, dz, kd, wd = pix2pix_data.BindName(half_time, ".npy")
        qpe = "qpepre_" + now_time + "-" + next_time + "_1_h.npy"
        if dr in DRList and dz in DZList and kd in KDList and qpe in preList:
            if wd in WDList and WDData:
                X.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd, WD + r"/" + wd])
            else:
                X.append([DR + r"/" + dr, DZ + r"/" + dz, KD + r"/" + kd])
            Y.append([qpepre + r"/" + qpe, 0])
        else:
            print("Not find " + now_time)

    return X, Y, timelist[:-1]


def op():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_h", type=int, default=64, help="size of model height")
    parser.add_argument("--model_w", type=int, default=64, help="size of model width")
    parser.add_argument("--img_h", type=int, default=64, help="size of really image height")
    parser.add_argument("--img_w", type=int, default=64, help="size of really image width")
    parser.add_argument("--epoch_num", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=1e-5, help="adam: learning rate")
    parser.add_argument("--batch_size_train", type=int, default=12, help="size of the training batches")
    parser.add_argument("--batch_size_val", type=int, default=1, help="size of the val batches")
    parser.add_argument("--batch_size_test", type=int, default=1, help="size of the test batches")
    parser.add_argument("--train_size", type=float, default=0.85, help="size of the training data")
    parser.add_argument("--DataAugmentation", type=bool, default=False, help="whether use the Data Augmentation")
    parser.add_argument("--AugmentationThreshold", type=int, default=70, help="")
    parser.add_argument("--Threshold", type=int, default=-1, help="")
    parser.add_argument("--Threshold_Avg", type=int, default=90, help="")
    parser.add_argument("--CSI_Threshold", type=int, default=5, help="")
    parser.add_argument("--Train", type=int, default=1, help="1 is Train, 0 is Test")
    parser.add_argument("--TestData", type=str, default="202105", help="")
    parser.add_argument("--pltSource", type=bool, default=False, help="")
    return parser.parse_args()


def main():
    # pix2pix_data.Generate_Hour_qpepre()
    # pix2pix_data.ProcessData()
    print(pix2pix_data.Region + " start *************************************************************")
    opt = op()
    opt.model_h, opt.model_w = pix2pix_data.Get_LongLat(Region, size=True)
    opt.img_h, opt.img_w = pix2pix_data.Get_LongLat(Region, size=True)
    if Region == "RS_TW":
        opt.img_h, opt.img_w = pix2pix_data.Get_LongLat("TW", size=True)
    model = pix2pixModel(opt)
    # model.predict_point(121.442, 24.998, start="2021/05/28 00:00", end="2021/05/31 19:00")
    # return

    X_train, Y_train, X_test, Y_test, X_val, Y_val = CreateDataSet(opt)

    print("train data: ", end="")
    DataLoader_train = DataLoader(NNDataset(X_train, Y_train), batch_size=opt.batch_size_train, num_workers=4)
    print("val data: ", end="")
    DataLoader_val = DataLoader(NNDataset(X_val, Y_val), batch_size=opt.batch_size_val, num_workers=4)
    # model.trainAuto(DataLoader_train, DataLoader_val)
    model.trainUNet(DataLoader_train, DataLoader_val)
    model.train(DataLoader_train, DataLoader_val)

    print("test data: ", end="")
    DataLoader_test = DataLoader(NNDataset(X_test, Y_test, True), batch_size=opt.batch_size_test, num_workers=1)
    model.testUnet(DataLoader_test)
    model.test(DataLoader_test)

    # model.predict_point(121.9153, 25.1308, start="2021/05/29 12:00", end="2021/05/31 12:00")

    print("finish")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # print("device_count = ", torch.cuda.device_count())
    main()
