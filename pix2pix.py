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
import pix2pix_data

seed = 164
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

SourcePath = r"./RadarData"
Path_QPE = r"./QPE"
Path_Pix2pix = r"./QPE/pix2pix"
Path_shp = r'./RadarData/shp/TWN_CITY.shp'


# SourcePath = r"E:/TWCC/RadarData"
# SourcePath = r"D:/Data2"
# Path_QPE = r"E:/TWCC/QPE"
# Path_Pix2pix = r"E:/TWCC/QPE/pix2pix"
# Path_shp = r"E:/TWCC/RadarData/shp/TWN_CITY.shp"


class NNDataset(Dataset):
    def __init__(self, X, Y, GetYFileName=False):
        self.X = X
        self.Y = Y
        self.w = 192
        self.h = 256
        self.Path = Path_QPE
        self.GetYFileName = GetYFileName
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

        if self.GetYFileName:
            return torch.from_numpy(X_).float(), torch.from_numpy(Y_), X_time, self.Y[i][0]
        else:
            return torch.from_numpy(X_).float(), torch.from_numpy(Y_).float().unsqueeze(0)

    def Preprocess(self, array, ci):
        # wissdom_out_Taiwan_mosaic_202105311730.npy
        if ci[-42:-17] == "wissdom_out_Taiwan_mosaic":
            return array[:, :, ::-1]
        return array

    def Y_Preprocess(self, Y_path):
        if not os.path.exists(Y_path):
            return np.zeros(1)
        array = np.load(Y_path)
        array = cv2.resize(array, (self.w, self.h))
        array = np.log2(array + 1) / np.log2(pix2pix_data.MaxPre + 1)
        return array


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.mean((outputs - targets) ** 2 * (targets / 20 + 1))


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
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class pix2pixModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.generator = None
        self.discriminator = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.lr = opt.lr
        self.loss = nn.MSELoss()
        self.L1loss = nn.MSELoss()
        self.customLoss = customLoss()
        self.cmap = None
        self.norm = None
        self.bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        self.epoch_num = opt.epoch_num
        self.threshold = opt.CSI_Threshold

    def train(self, DataLoader_train, DataLoader_val):
        self.load_model(False)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr / 20, betas=(0.5, 0.999))

        lessLoss = None
        count = 0
        log_train, log_val = [], []
        need_frozen_list = ['module.conv_hPA3.weight', 'module.conv_hPA3.bias',
                            'module.conv_hPA3_2.weight', 'module.conv_hPA3_2.bias',
                            'module.conv_hPA3_3.weight', 'module.conv_hPA3_3.bias']
        for epoch in range(1, self.epoch_num + 1):
            NowTime = time()
            loss_D_train = 0
            loss_GAN_train = 0
            loss_pixel_train = 0

            for i, (train_x, train_y) in enumerate(DataLoader_train, 1):
                self.generator.train()
                self.discriminator.train()
                if epoch > 50:
                    if train_x.shape[3] == 24 + 48:  # frozen
                        for param in self.generator.named_parameters():
                            if param[0] not in need_frozen_list:
                                param[1].requires_grad = False
                    elif train_x.shape[3] == 24:
                        for param in self.generator.named_parameters():
                            param[1].requires_grad = True
                loss_D, loss_G, loss_pixel, loss_GAN = self.train_step(train_x, train_y.to(device))
                loss_D_train += loss_D
                loss_GAN_train += loss_GAN
                loss_pixel_train += loss_pixel

            if epoch % 1 == 0:
                self.generator.eval()
                self.discriminator.eval()
                with torch.no_grad():
                    loss_D_val = 0
                    loss_pixel_val = 0
                    for i, (real_A, real_B) in enumerate(DataLoader_val):
                        real_B = real_B.to(device)
                        Tensor = torch.cuda.FloatTensor
                        patch = (1, self.opt.img_height // 2 ** 4, self.opt.img_width // 2 ** 4)
                        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
                        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
                        fake_B = self.generator(real_A)
                        # Pixel-wise loss
                        loss_pixel = self.customLoss(fake_B, real_B)
                        loss_G = loss_pixel

                        # Real loss
                        pred_real = self.discriminator(real_B, real_B)
                        loss_real = self.L1loss(pred_real, valid)
                        # Fake loss
                        pred_fake = self.discriminator(fake_B.detach(), real_B)
                        loss_fake = self.L1loss(pred_fake, fake)
                        # Total loss
                        loss_D = (loss_fake + loss_real) / 2

                        loss_D_val += loss_D.item()
                        loss_pixel_val += loss_G.item()

                loss_D_train *= 1000
                loss_GAN_train *= 1000
                loss_pixel_train *= 1000
                loss_D_val *= 1000
                loss_pixel_val *= 1000
                print(
                    "[Epoch %d, duration=%.2f] [train D loss: %f, GAN loss: %f, pixel loss: %f] "
                    "[val D loss: %f, pixel loss: %f]"
                    % (
                        epoch, (time() - NowTime) / 60,
                        loss_D_train / len(DataLoader_train),
                        loss_GAN_train / len(DataLoader_train), loss_pixel_train / len(DataLoader_train),
                        loss_D_val / len(DataLoader_val), loss_pixel_val / len(DataLoader_val)
                    )
                )
                if lessLoss is None or loss_pixel_val < lessLoss:
                    lessLoss = loss_pixel_val
                    count = 0
                    self.save_model()
                else:
                    count += 1
                    if count == 1 or count == 6:
                        self.lr *= 0.9
                        print("lr = ", self.lr)
                        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
                        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr / 20)
                log_train.append(loss_pixel_train / len(DataLoader_train))
                log_val.append(loss_pixel_val / len(DataLoader_val))
                try:
                    plt.plot(log_train, label="train")
                    plt.plot(log_val, label="val")
                    plt.legend()
                    plt.xlabel("epoch")
                    plt.ylim(1.5, 5.5)
                    plt.ylabel("loss")
                    plt.grid(linestyle="--", alpha=0.3)
                    plt.savefig(Path_Pix2pix + "/loss.png")
                    plt.close()
                except:
                    plt.close()
                if count > 10:
                    print("break epoch = ", epoch)
                    break

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor):
        Tensor = torch.cuda.FloatTensor
        patch = (1, self.opt.img_height // 2 ** 4, self.opt.img_width // 2 ** 4)
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)  # torch.Size([2, 1, 16, 12])
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        #  Train Generators
        self.optimizer_G.zero_grad()  # zero the gradient buffers
        # GAN loss
        fake_B = self.generator(real_A)
        pred_fake = self.discriminator(fake_B, real_B)
        loss_GAN = self.L1loss(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = self.customLoss(fake_B, real_B)
        weight = 100
        loss_G = loss_GAN + weight * loss_pixel
        loss_G.backward()
        # print(self.generator.module.down1_wd.model[0].weight.grad)
        self.optimizer_G.step()

        #  Train Discriminator
        self.optimizer_D.zero_grad()
        # Real loss
        pred_real = self.discriminator(real_B, real_B)
        loss_real = self.L1loss(pred_real, valid)
        # Fake loss
        pred_fake = self.discriminator(fake_B.detach(), real_B)
        loss_fake = self.L1loss(pred_fake, fake)
        # Total loss
        loss_D = (loss_fake + loss_real) / 2
        loss_D.backward()
        self.optimizer_D.step()

        return loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item()

    def pltconfig(self):
        self.cmap = mpl.colors.ListedColormap(['lightcyan', 'cyan', 'cornflowerblue', 'blue',
                                               'lime', 'limegreen', 'green', 'yellow', 'orange',
                                               'red', 'tab:red', 'brown', 'fuchsia', 'blueviolet'])
        self.cmap.set_under('white')
        self.cmap.set_over('lavender')
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)

    def plt_shp(self, img, filename):
        if self.cmap is None:
            self.pltconfig()
        if not os.path.exists(Path_shp):
            print("please upload the shp file at ./shp/TWN_CITY.shp")
            return
        TWN_CITY = gpd.read_file(Path_shp)
        x = np.arange(120, 122.0125, 0.0125)
        y = np.arange(21.8875, 25.3125, 0.0125)
        x, y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        bounds = self.bounds
        norm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)
        plt.contourf(x, y, img[::-1, :], bounds, cmap=self.cmap)
        ax = TWN_CITY.geometry.plot(ax=ax, alpha=0.3)
        plt.xlim(120, 122.0125)
        plt.ylim(21.8875, 25.3125)
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
        site = np.load(os.path.join(SourcePath, "site.npy"))
        gt, p = gt[site == 1], p[site == 1]

        mer = 100 * (np.abs(gt - p) / (gt + 1.01)).mean()
        csi_and = np.logical_and(gt > threshold, p > threshold).sum()
        csi_or = np.logical_or(gt > threshold, p > threshold).sum()
        csi = csi_and / csi_or if csi_or != 0 else None
        mse = np.square(np.subtract(gt, p)).mean()
        str = "gt, avg=%.5f, max=%.3f, min=%.3f \np , avg=%.5f, max=%.3f, min=%.3f \n" \
              % (np.mean(gt), np.max(gt), np.min(gt), np.mean(p), np.max(p), np.min(p))
        return mer, csi, str, csi_and, csi_or, mse

    def cal(self, gt, p):
        if os.path.exists(os.path.join(SourcePath, "site.npy")):
            return self.cal_site(gt, p)

        threshold = self.threshold
        mer = 100 * (np.abs(gt - p) / (gt + 1.01)).mean()
        csi_and = np.logical_and(gt > threshold, p > threshold).sum()
        csi_or = np.logical_or(gt > threshold, p > threshold).sum()
        csi = csi_and / csi_or if csi_or != 0 else None
        mse = np.square(np.subtract(gt, p)).mean()
        str = "gt, avg=%.5f, max=%.3f, min=%.3f \np , avg=%.5f, max=%.3f, min=%.3f \n" \
              % (np.mean(gt), np.max(gt), np.min(gt), np.mean(p), np.max(p), np.min(p))
        return mer, csi, str, csi_and, csi_or, mse

    def test(self, DataLoader_test):
        print("pix2pixModel test")
        self.load_model(True)
        if not os.path.exists(Path_Pix2pix):
            os.mkdir(Path_Pix2pix)
        doc_test = Path_Pix2pix + "/test" + datetime.datetime.today().strftime("_%m%d_%H%M")
        if not os.path.exists(doc_test):
            os.mkdir(doc_test)
        self.pltconfig()
        if os.path.exists(os.path.join(SourcePath, "site.npy")):
            print("test by site.npy")
        with open(doc_test + "/cal.txt", "w") as f:
            f.writelines(Path_Pix2pix + "\n")
        temp_mer, temp_mse, temp_csi, temp_avg, temp_max = [], [], [], [], []
        sum_csi_and, sum_csi_or, sum_csi_and_avg, sum_csi_or_avg, sum_csi_and_max, sum_csi_or_max = 0, 0, 0, 0, 0, 0

        self.generator.eval()
        with torch.no_grad():
            loss_test = 0
            for _, (test_x, test_y, X_time, YName) in enumerate(DataLoader_test):
                output = self.generator(test_x).squeeze(1)
                loss_test += self.loss(output, test_y.float().to(device)).item()
                for b in range(test_x.shape[0]):
                    p = output[b, :, :].cpu().numpy().reshape(self.opt.img_height, self.opt.img_width)
                    p = cv2.resize(2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1, (162, 275))
                    doc = doc_test + "/" + YName[b][-40:-4]  # qpepre_202105010600-202105010700_1_h
                    if not os.path.exists(doc):
                        os.mkdir(doc)
                    self.plt_shp(p, doc + "/p" + ".png")
                    self.writefile(p, doc + "/" + YName[b][-40:-4] + ".txt")
                    if self.opt.pltSource:
                        for t in X_time:
                            self.pltRadar(t[0], doc)
                    # 存在 label
                    if os.path.exists(YName[b]):
                        gt = np.load(YName[b])
                        self.plt_shp(gt, doc + "/gt" + ".png")
                        mer, csi, log_str, csi_and, csi_or, mse = self.cal(gt, p)
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
                        sum_csi_and += csi_and
                        sum_csi_or += csi_or
                        if max_gt >= 40:
                            sum_csi_and_max += csi_and
                            sum_csi_or_max += csi_or
                        if mean_gt > 1:
                            sum_csi_and_avg += csi_and
                            sum_csi_or_avg += csi_or

                        with open(doc_test + "/cal.txt", "a") as f:
                            cal_log = "%s : MER=%.5f, MSE=%.3f, " % (YName[b][-40:-4], mer, mse) \
                                      + ("CSI=NaN" if csi is None else ("CSI=%.3f" % csi))
                            print(cal_log)
                            f.writelines(cal_log + "\n")
                            f.writelines(log_str)
        if len(temp_mer) == 0:
            return
        loss_test *= 1000
        # print("DataLoader_test: ", loss_test / len(DataLoader_test))
        with open(doc_test + "/cal.txt", "a") as f:
            if sum_csi_or == 0:
                sum_csi_or = 1
            if sum_csi_or_max == 0:
                sum_csi_or_max = 1
            if sum_csi_or_avg == 0:
                sum_csi_or_avg = 1
            f.writelines("Sum : MER=%.5f, CSI=%.3f, MSE=%.3f  \n" %
                         (sum(temp_mer) / len(temp_mer), sum_csi_and / sum_csi_or, sum(temp_mse) / len(temp_mse)))
            f.writelines("topcsi_avg=%.3f, topcsi_max=%.3f \n" % (sum_csi_and_avg / sum_csi_or_avg,
                                                                  sum_csi_and_max / sum_csi_or_max))
            topmer_avg = list(
                filter(lambda x: list(map(lambda te: float(te) > 1, temp_avg))[temp_mer.index(x)] is True, temp_mer))
            topmer_max = list(
                filter(lambda x: list(map(lambda te: int(te) >= 40, temp_max))[temp_mer.index(x)] is True, temp_mer))
            f.writelines("topmer_avg=%.3f, topmer_max=%.3f \n" % (sum(topmer_avg) / len(topmer_avg),
                                                                  sum(topmer_max) / len(topmer_max)))
            topmse_avg = list(
                filter(lambda x: list(map(lambda te: float(te) > 1, temp_avg))[temp_mse.index(x)] is True, temp_mse))
            topmse_max = list(
                filter(lambda x: list(map(lambda te: int(te) >= 40, temp_max))[temp_mse.index(x)] is True, temp_mse))
            f.writelines("topmse_avg=%.3f, topmse_max=%.3f \n" % (sum(topmse_avg) / len(topmse_avg),
                                                                  sum(topmse_max) / len(topmse_max)))

        print("Mean error rate = %.5f " % (sum(temp_mer) / len(temp_mer)) + "%")
        print("Critical Success Index = %.5f " % (sum_csi_and / sum_csi_or))
        print("Mean square error = %.5f " % (sum(temp_mse) / len(temp_mse)))

        # 繪圖
        plt.hist(temp_mer)
        plt.xlabel("Mean error rate (%)")
        plt.ylabel("Count")
        plt.savefig(doc_test + "/mer.png")
        plt.close()

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

        plt.scatter(temp_avg, temp_mer, s=30, c='blue', marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
        plt.xlabel('Average rain fall (mm/hr)')
        plt.ylabel('Mean error rate (%)')
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(doc_test + "/avg_mer.png")
        plt.close()

        plt.scatter(temp_max, temp_mer, s=30, c='blue', marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
        plt.xlabel('Maximum rain fall (mm/hr)')
        plt.ylabel('Mean error rate (%)')
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(doc_test + "/max_mer.png")
        plt.close()

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

    def pltRadar(self, file, doc_pltRadar=""):
        if not os.path.exists(Path_shp):
            print("please upload the shp file at ./shp/TWN_CITY.shp")
            return

        time = file[-16:-4]
        type = ""
        for t in ["DR", "DZ", "KD", "WD"]:
            if t in file:
                type = t

        dic = {"DR": "ZDR", "DZ": "ZH", "KD": "KDP", "WD": "WISSDOM"}

        if type == "WD":
            (img_u, img_v, img_w) = pix2pix_data.SourceToNpy(pix2pix_data.TimeToSourcePath(time, type), type)
            if img_u is None:
                print(time, type, "is None")
                return
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
                plt.contourf(x, y, img[::-1, :, 8 - i], level_bound, cmap=cmap)
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

    def load_model(self, exist):
        self.generator = nn.DataParallel(GeneratorUNet().to(device))
        self.discriminator = nn.DataParallel(Discriminator().to(device))
        if exist and os.path.exists(Path_Pix2pix):
            self.generator.module.load_state_dict(torch.load(Path_Pix2pix + "/generator.pth"))
            self.discriminator.module.load_state_dict(torch.load(Path_Pix2pix + "/discriminator.pth"))

    def predict_point(self, long=120.2420, lat=24.5976, start="2021/05/28 00:00", end="2021/05/31 19:00"):
        X, Y, timelist = CreateContinuousData(start, end)
        gt_point, p_point = [], []
        i, j = -int((lat - 21.8875) // 0.0125) - 1, int((long - 120) // 0.0125)  # 座標點
        print("predict data: ", end="")
        predict_data = DataLoader(NNDataset(X, Y, True), batch_size=1, num_workers=1)
        self.load_model(True)
        with torch.no_grad():
            for _, (test_x, _, _, YName) in enumerate(predict_data):
                output = self.generator(test_x).squeeze(1)
                for b in range(test_x.shape[0]):
                    p = output[b, :, :].cpu().numpy().reshape(self.opt.img_height, self.opt.img_width)
                    p = cv2.resize(2 ** (p * np.log2(pix2pix_data.MaxPre + 1)) - 1, (162, 275))
                    gt = np.load(YName[b])
                    p_point.append(p[i][j])
                    gt_point.append(gt[i][j])

        with open(os.path.join(Path_Pix2pix, "predict_%.3f°E_%.3f°N.txt" % (long, lat)), "w") as f:
            for i in range(len(p_point)):
                value = ("%.2f" % p_point[i]).rjust(9, " ")
                f.writelines("%s  %3.4f   %2.4f" % (datetime.datetime.strftime(timelist[i], "%Y/%m/%d %H:%M"),
                                                    long, lat) + value + "\n")

        if len(timelist) < 24 * 4:
            sample_tick = [t for t in timelist if t.hour in [0, 6, 12, 18]]
            tick = [datetime.datetime.strftime(t, "%Y/%m/%d %H:%M") for t in sample_tick]
            tick_label = [datetime.datetime.strftime(t, "%m/%d") if t.hour == 0 else str(t.hour) for t in sample_tick]
        else:
            sample_tick = timelist[::int(len(timelist) / 6)]
            tick = [t + " 00:00" for t in [datetime.datetime.strftime(t, "%Y/%m/%d") for t in sample_tick]]
            tick_label = [datetime.datetime.strftime(t, "%m/%d") for t in sample_tick]
        plotlist = [datetime.datetime.strftime(t, "%Y/%m/%d %H:%M") for t in timelist]
        plt.plot(plotlist, gt_point, label="gt")
        plt.plot(plotlist, p_point, label="p")
        plt.xticks(tick, tick_label)
        plt.xlabel(start + " ~ " + end)
        plt.ylabel("mm/hr")
        plt.legend()
        plt.title("%.3f°E %.3f°N" % (long, lat))
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(os.path.join(Path_Pix2pix, "predict_%.3f°E_%.3f°N.png" % (long, lat)))


def CreateDataSet(opt, WDData=True):
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

    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/Z", Path_QPE + r"/KD", Path_QPE + r"/WD"
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
    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/Z", Path_QPE + r"/KD", Path_QPE + r"/WD"
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), \
                                              os.listdir(KD), os.listdir(WD)
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
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=192, help="size of image width")
    parser.add_argument("--epoch_num", type=int, default=250, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
    parser.add_argument("--batch_size_train", type=int, default=7, help="size of the training batches")
    parser.add_argument("--batch_size_val", type=int, default=7, help="size of the val batches")
    parser.add_argument("--batch_size_test", type=int, default=1, help="size of the test batches")
    parser.add_argument("--train_size", type=float, default=0.85, help="size of the training data")
    parser.add_argument("--DataAugmentation", type=bool, default=True, help="whether use the Data Augmentation")
    parser.add_argument("--AugmentationThreshold", type=int, default=70, help="")
    parser.add_argument("--Threshold", type=int, default=15, help="")
    parser.add_argument("--Threshold_Avg", type=int, default=90, help="")
    parser.add_argument("--CSI_Threshold", type=int, default=10, help="")
    parser.add_argument("--Train", type=bool, default=True, help="")
    parser.add_argument("--TestData", type=str, default="202105", help="")
    parser.add_argument("--pltSource", type=bool, default=False, help="")
    return parser.parse_args()


def main():
    pix2pix_data.ProcessData()
    print("start **********************************************************************************")
    opt = op()
    model = pix2pixModel(opt)
    # model.predict_point(long=120.2420, lat=24.5976, start="2021/05/28 00:00", end="2021/05/31 19:00")

    if opt.Train:
        X_train, Y_train, X_test, Y_test, X_val, Y_val = CreateDataSet(opt)
        print("train data: ", end="")
        DataLoader_train = DataLoader(NNDataset(X_train, Y_train), batch_size=opt.batch_size_train, num_workers=4)
        print("val data: ", end="")
        DataLoader_val = DataLoader(NNDataset(X_val, Y_val), batch_size=opt.batch_size_val, num_workers=4)
        model.train(DataLoader_train, DataLoader_val)
    else:
        X_test, Y_test = CreateTestData(opt)
    print("test data: ", end="")
    DataLoader_test = DataLoader(NNDataset(X_test, Y_test, True), batch_size=opt.batch_size_test, num_workers=1)
    model.test(DataLoader_test)

    print("finish")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # print("device_count = ", torch.cuda.device_count())
    main()
