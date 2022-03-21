import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import warnings
import struct
import array as arr
import sys
import traceback
import datetime
import shutil
import pandas as pd

warnings.filterwarnings('ignore')

doc = r"E:/TWCC3"  # NT, ST, TWCC
Region = "RS_TW"  # NT, ST, RS_TW
# ===================================================================
SourcePath = doc + r"/RadarData"
SourcePath = r"D:/Data2"
Path_QPE = doc + r"/QPE"
Path_Pix2pix = doc + r"/QPE/pix2pix"
Path_shp = SourcePath + r"/shp/TWN_CITY.shp"
# MaxDR, MinDR, MaxDZ, MinDZ, MaxKD, MinKD, MaxPre, MinPre, MaxZ, MinZ = \
#     25.496891, 0.0, 110.35679, -30.603172, 39.522815, 0.0, 99.75, 0.0, 10 ** 11.035679, 10 ** -3.0603172
# MaxWD_u, MinWD_u, MaxWD_v, MinWD_v, MaxWD_w, MinWD_w, = \
#     26.52743215445662, -21.127624244138133, 30.38607470842544, -25.399568055872805, \
#     11.52811917316285, -6.693125475125271
# MaxDR, MinDR, MaxDZ, MinDZ, MaxKD, MinKD, MaxPre, MinPre = 13.6, 0, 86, -25.4, 168, 0, 99, 0.0
MaxDR, MinDR, MaxDZ, MinDZ, MaxKD, MinKD, MaxPre, MinPre = 7.9, 0, 86, -25.4, 10, 0, 99, 0.0
# MaxDR, MinDR, MaxDZ, MinDZ, MaxKD, MinKD, MaxPre, MinPre = \
#     1.001225 + 4 * (1.305410 - 0.5669985), 1.001225 - 4 * (1.305410 - 0.5669985), \
#     20.07065 + 4 * (27.94392 - 11.99222), 20.07065 - 4 * (27.94392 - 11.99222), \
#     0.818381 + 4 * (0.6719755 - 0.2101978), 0.818381 - 4 * (0.6719755 - 0.2101978), 99, 0.0
# 3.954871 -1.952421 83.87745 -43.736149999999995 2.6654918 -1.0287297999999998 99 0.0
MaxWD_u, MinWD_u, MaxWD_v, MinWD_v, MaxWD_w, MinWD_w, = \
    2.536048 + 4 * (5.509512 - -0.5780655), 2.536048 - 4 * (5.509512 - -0.5780655), \
    1.502947 + 4 * (3.836798 - -0.5885332), 1.502947 - 4 * (3.836798 - -0.5885332), \
    0.009922518 + 4 * (0.1160055 - -0.09846296), 0.009922518 - 4 * (0.1160055 - -0.09846296)
# 26.886358 -21.814262 19.204271799999997 -16.1983778 0.867796358 -0.847951322


def medianBlur():
    def unpack(sourPath, Type):
        def DataToImg(data):
            c = str(501 * 381 * 21)
            img = np.array(struct.unpack(c + 'f', data)).reshape(21, 501, 381).transpose(1, 2, 0)[::-1, :, :]
            return img

        if Type == "WD":
            with open(sourPath, "rb") as f:
                data = f.read()
            img_u = DataToImg(data[6 * 381 * 501 * 21 * 4: 7 * 381 * 501 * 21 * 4])
            img_v = DataToImg(data[7 * 381 * 501 * 21 * 4: 8 * 381 * 501 * 21 * 4])
            img_w = DataToImg(data[8 * 381 * 501 * 21 * 4: 9 * 381 * 501 * 21 * 4])
            return img_u, img_v, img_w
        elif Type in ["DR", "DZ", "KD"]:
            with open(sourPath, "rb") as f:
                float_array = arr.array('f')
                float_array.fromfile(f, 441 * 561 * 34)
                array = np.array(float_array).reshape(-1).reshape(34, 561, 441).transpose(1, 2, 0)[::-1, :, :]
                if Type == "DZ":
                    array = np.where(array == -9999, MinDZ, array)
                else:
                    array = np.where(array == -9999, 0, array)
                return array
        elif Type == "qpepre":
            npy = []
            temp = 0
            with open(sourPath, "r") as f:
                lines = f.readline()
                while lines:
                    if lines[19:26] != temp:
                        npy.append([])
                        temp = lines[19:26]
                    npy[-1].append(float(lines[30:35]))
                    lines = f.readline()
            array = np.array(npy[::-1])
            return array

    radar = SplitRegion(unpack(os.path.join("D:\Data2", "CAPPI_COMP_DR_202005250830.bin"), "DZ"), "Radar", "TW")
    # cv2.medianBlur
    plt.imshow(radar)
    plt.show()


def ViewTest(docList):
    VTdoc = os.path.join(Path_Pix2pix, "ViewTest")
    if not os.path.exists(VTdoc):
        os.mkdir(VTdoc)

    # 轉為絕對路徑
    docList = [os.path.join(Path_Pix2pix, "test_" + i) for i in docList]

    for test in os.listdir(docList[0]):
        # 是否為個案資料夾
        if os.path.isdir(os.path.join(docList[0], test)):
            exists = True
            # 是否皆存在此個案
            for d in docList:
                if not os.path.exists(os.path.join(d, test)):
                    exists = False
            if exists:
                print(test)
                os.mkdir(os.path.join(Path_Pix2pix, "ViewTest", test))
                for d in docList:
                    with open(os.path.join(d, test, "cal.txt")) as f:
                        l = f.readline()
                        mse = "_" + l[l.index("MSE=") + 4: l.index("MSE=") + 8]
                    shutil.copyfile(os.path.join(d, test, "p.png"), os.path.join(VTdoc, test, d[-9:] + mse + ".png"))
                shutil.copyfile(os.path.join(d, test, "gt.png"), os.path.join(VTdoc, test, "gt.png"))


def box_Data(tp):
    def unpack(sourPath, Type):
        def DataToImg(data):
            c = str(501 * 381 * 21)
            img = np.array(struct.unpack(c + 'f', data)).reshape(21, 501, 381).transpose(1, 2, 0)[::-1, :, :]
            return img

        if Type == "WD":
            with open(sourPath, "rb") as f:
                data = f.read()
            img_u = DataToImg(data[6 * 381 * 501 * 21 * 4: 7 * 381 * 501 * 21 * 4])
            img_v = DataToImg(data[7 * 381 * 501 * 21 * 4: 8 * 381 * 501 * 21 * 4])
            img_w = DataToImg(data[8 * 381 * 501 * 21 * 4: 9 * 381 * 501 * 21 * 4])
            return img_u, img_v, img_w
        elif Type in ["DR", "DZ", "KD"]:
            with open(sourPath, "rb") as f:
                float_array = arr.array('f')
                float_array.fromfile(f, 441 * 561 * 34)
                array = np.array(float_array).reshape(-1).reshape(34, 561, 441).transpose(1, 2, 0)[::-1, :, :]
                # if Type == "DZ":
                #     array = np.where(array == -9999, MinDZ, array)
                # else:
                #     array = np.where(array == -9999, 0, array)
                return array
        elif Type == "qpepre":
            npy = []
            temp = 0
            with open(sourPath, "r") as f:
                lines = f.readline()
                while lines:
                    if lines[19:26] != temp:
                        npy.append([])
                        temp = lines[19:26]
                    npy[-1].append(float(lines[30:35]))
                    lines = f.readline()
            array = np.array(npy[::-1])
            return array

    nums, nums_uvw = [], [[] for i in range(3)]
    i = 0
    # qpepre_202005010000-202005010100_1_h
    # CAPPI_COMP_DZ_202005010000 DR KD
    # wissdom_out_Taiwan_mosaic_202105020030
    for dirPath, dirNames, fileNames in os.walk(SourcePath):
        # print(dirPath)
        for file in fileNames:
            if file[-4:] == ".bin":
                Type = file[11:13]
                year = file[14:18]
                month = file[18:20]
                if file[:7] == "wissdom":
                    Type = "WD"
                    year = file[26:30]
                    month = file[30:32]
                if Type == tp:
                    i += 1
                    if Type in ["DR", "DZ", "KD"]:
                        img = unpack(os.path.join(dirPath, file), Type)
                        img = SplitRegion(img, Type, "TW")
                        img = img[:, :, 1:9]
                        # img = np.concatenate((img[:, :, 2:7], img[:, :, 28:31]), 2)
                        assert img.shape[2] == 8
                        img = img[img != -9999]
                        if tp == "KD":
                            # img = img[img > 2.5]
                            img = img[img < 10]
                        if tp == "DZ":
                            img = img[img > (-30)]
                        if tp == "DR":
                            # img = img[img > 1.8]
                            img = img[img < 7.9]
                        # img = img[img != 0]
                        l = img.shape[0]
                        indices = np.random.choice(l, int(l * 0.05), replace=False)
                        nums += list(img[indices])
                    if Type == "WD":
                        img_u, img_v, img_w = unpack(os.path.join(dirPath, file), Type)
                        for uvw, img in enumerate([img_u, img_v, img_w]):
                            img = SplitRegion(img, "WD", "TW")
                            img = img[:, :, 1:9]
                            assert img.shape[2] == 8
                            img = img[img != -999]
                            l = img.shape[0]
                            indices = np.random.choice(l, int(l * 0.05), replace=False)
                            nums_uvw[uvw] += list(img[indices])

    if tp in ["DR", "DZ", "KD"]:
        df = pd.DataFrame(nums)
        print(tp, df.describe())
        df.plot.box(title=tp)
        plt.grid(linestyle="--", alpha=0.3)
        plt.savefig(r"E:/TWCC/QPE/Research" + tp + "_new2.png")
        plt.figure()
        np.save(r"E:/TWCC/QPE/Research" + tp + "_new2.npy", np.array(nums))
    if tp == "WD":
        word = ["U", "V", "W"]
        for uvw, nums in enumerate(nums_uvw):
            df = pd.DataFrame(nums)
            print("WD_", word[uvw], df.describe())
            df.plot.box(title=tp)
            plt.grid(linestyle="--", alpha=0.3)
            plt.savefig(r"E:/TWCC/QPE/ResearchWD_" + word[uvw] + "_new2.png")
            plt.figure()
            np.save(r"E:/TWCC/QPE/ResearchWD_" + word[uvw] + "_new2.npy", np.array(nums))

    print(tp + " final, i = ", i)


def Pearson_Correlation():
    def unpack(sourPath, Type):
        def DataToImg(data):
            c = str(501 * 381 * 21)
            img = np.array(struct.unpack(c + 'f', data)).reshape(21, 501, 381).transpose(1, 2, 0)[::-1, :, :]
            return img

        if Type == "WD":
            with open(sourPath, "rb") as f:
                data = f.read()
            img_u = DataToImg(data[6 * 381 * 501 * 21 * 4: 7 * 381 * 501 * 21 * 4])
            img_v = DataToImg(data[7 * 381 * 501 * 21 * 4: 8 * 381 * 501 * 21 * 4])
            img_w = DataToImg(data[8 * 381 * 501 * 21 * 4: 9 * 381 * 501 * 21 * 4])
            return img_u, img_v, img_w
        elif Type in ["DR", "DZ", "KD"]:
            with open(sourPath, "rb") as f:
                float_array = arr.array('f')
                float_array.fromfile(f, 441 * 561 * 34)
                array = np.array(float_array).reshape(-1).reshape(34, 561, 441).transpose(1, 2, 0)[::-1, :, :]
                if Type == "DZ":
                    array = np.where(array == -9999, MinDZ, array)
                else:
                    array = np.where(array == -9999, 0, array)
                return array
        elif Type == "qpepre":
            npy = []
            temp = 0
            with open(sourPath, "r") as f:
                lines = f.readline()
                while lines:
                    if lines[19:26] != temp:
                        npy.append([])
                        temp = lines[19:26]
                    npy[-1].append(float(lines[30:35]))
                    lines = f.readline()
            array = np.array(npy[::-1])
            return array

    hh = 34
    site = np.load(os.path.join(SourcePath, "site.npy"))
    # print(np.sum(site))
    # site = SplitRegion(site, "TW", Region)
    Corr_gt = []
    Corr_radar = [[] for i in range(hh)]
    filelist = []

    for dirPath, dirNames, fileNames in os.walk(SourcePath):
        print(dirPath)
        for file in fileNames:
            if file[-4:] == ".txt":
                # 儲存.npy
                try:
                    if file not in filelist:
                        filelist.append(file)
                    else:
                        continue
                    gt = unpack(os.path.join(dirPath, file), "qpepre")[site == 1]

                    b, e = file[7:19], file[20:32]
                    year = b[:4]
                    if year != "2021":
                        continue
                    h = datetime.datetime.strftime(datetime.datetime.strptime(b, "%Y%m%d%H%M")
                                                   + datetime.timedelta(minutes=30), "%Y%m%d%H%M")
                    dr, dz, kd, wd = BindName(h, ".bin")
                    for dirPath2, dirNames, fileNames in os.walk(SourcePath):
                        for file in fileNames:
                            if file == dr:
                                print(file)
                                img_u, img_v, img_w = unpack(os.path.join(dirPath2, file), "WD")
                                img = img_w
                                img = SplitRegion(img, "WD", "TW")
                                img = np.where(img == -999, 0, img)
                                h, w = Get_LongLat("TW", True)
                                radar = cv2.resize(img, (w, h))
                                # radar = np.where(radar == -9999, 0, radar)
                                Corr_gt = Corr_gt + list(gt)
                                # Corr_gt = Corr_gt + list(gt[gt >= 10])
                                for i in range(hh):
                                    radar_site = radar[:, :, i][site == 1]
                                    Corr_radar[i] = Corr_radar[i] + list(radar_site)
                                    # Corr_radar[i] = Corr_radar[i] + list(radar_site[gt >= 10])
                except Exception as e:
                    print("error file: ", file)
                    print(errMsg(e))
                    return

    from scipy.stats import pearsonr
    correlation, pvalue = [], []
    print("len: ", len(Corr_gt))
    print("len: ", len(Corr_radar[0]))
    for i in range(hh):
        c, p = pearsonr(Corr_gt, Corr_radar[i])
        correlation.append(c)
        pvalue.append(p)
        print(str(i + 1) + ": ", c, p)
    plt.bar(range(hh), correlation)
    # DR, KD, DZ
    # differential reflectivity, specific differential phase, the reflectivity
    plt.xlabel("DR")
    plt.ylabel("correlation")
    plt.savefig(r"D:/correlation.png")
    plt.close()
    plt.bar(range(hh), pvalue)
    plt.savefig(r"D:/pvalue.png")
    plt.close()

    # Generate_Hour_qpepre()
    print("Correlation final")


def ProcessData(dBZtoZ=True):
    def np_save(file, data):
        arr, msg = data
        if arr is None:
            print(msg)
        else:
            np.save(file, arr)

    def ProcessNpy(sourPath, Type, dBZtoZ=True):
        # 處理npy 和歸一化
        def ImgToWD(img_u, img_v, img_w):
            h, w = Get_LongLat("TW", True)
            img_u = cv2.resize(img_u, (w, h))
            img_v = cv2.resize(img_v, (w, h))
            img_w = cv2.resize(img_w, (w, h))
            img_u = np.where(img_u >= 0, np.log(img_u + 1) / np.log(MaxWD_u + 1),
                             -np.log(-img_u + 1) / np.log(-MinWD_u + 1))
            img_v = np.where(img_v >= 0, np.log(img_v + 1) / np.log(MaxWD_v + 1),
                             -np.log(-img_v + 1) / np.log(-MinWD_v + 1))
            img_w = np.where(img_w >= 0, np.log(img_w + 1) / np.log(MaxWD_w + 1),
                             -np.log(-img_w + 1) / np.log(-MinWD_w + 1))

            img = np.zeros((h, w, 48))
            for i in range(h):
                for j in range(w):
                    for k in range(8):
                        if img_u[i][j][k] >= 0:
                            img[i][j][k * 6 + 0] = img_u[i][j][k]
                        else:
                            img[i][j][k * 6 + 1] = -img_u[i][j][k]
                        if img_v[i][j][k] >= 0:
                            img[i][j][k * 6 + 2] = img_v[i][j][k]
                        else:
                            img[i][j][k * 6 + 3] = -img_v[i][j][k]
                        if img_w[i][j][k] >= 0:
                            img[i][j][k * 6 + 4] = img_w[i][j][k]
                        else:
                            img[i][j][k * 6 + 5] = -img_w[i][j][k]
            return img

        def Preprocess(array, Type):
            if Type == "DR":
                array = np.where(array > MaxDR, MaxDR, array)
            elif Type == "DZ":
                array = np.where(array > MaxDZ, MaxDZ, array)
            elif Type == "KD":
                array = np.where(array > MaxKD, MaxKD, array)

            # if Region == "RS_TW":
            #     array = cv2.resize(array, (192, 256))

            # dBZ to Z
            if Type == "Z":
                # None
                array = 10 ** (array / 10)
                array = np.log(array + 1 - MinZ) / np.log(MaxZ - MinZ + 1)
            elif Type == "DR":  # DR
                array = array / MaxDR
            elif Type == "DZ":  # DZ
                array = (array - MinDZ) / (MaxDZ - MinDZ)
            elif Type == "KD":  # KD
                array = array / MaxKD
            return array

        try:
            if Type == "WD":
                (img_u, img_v, img_w) = SourceToNpy(sourPath, Type, Split=True)
                return ImgToWD(img_u, img_v, img_w), ""
            elif Type in ["DR", "DZ", "KD"]:
                # dBZ to Z
                array = SourceToNpy(sourPath, Type, Split=True)
                if Type == "DZ" and dBZtoZ:
                    return Preprocess(array, "Z"), ""
                return Preprocess(array, Type), ""
            elif Type == "qpepre":
                return SourceToNpy(sourPath, Type, Split=True), ""
        except Exception as e:
            return None, errMsg(e)

    if not os.path.exists(Path_shp):
        raise Exception("please upload the shp file at ./shp/TWN_CITY.shp")

    def makedirs(Path_QPE):
        if not os.path.exists(Path_QPE):
            os.makedirs(Path_QPE)
        if not os.path.exists(Path_QPE + "/DR"):
            os.makedirs(Path_QPE + "/DR")
        if not os.path.exists(Path_QPE + "/DZ"):
            os.makedirs(Path_QPE + "/DZ")
        if not os.path.exists(Path_QPE + "/KD"):
            os.makedirs(Path_QPE + "/KD")
        if not os.path.exists(Path_QPE + "/WD"):
            os.makedirs(Path_QPE + "/WD")
        if not os.path.exists(Path_QPE + "/qpepre"):
            os.makedirs(Path_QPE + "/qpepre")
        # if not os.path.exists(Path_QPE + "/Z"):
        #     os.makedirs(Path_QPE + "/Z")

    makedirs(Path_QPE)
    StorePath = Path_QPE + "/"

    for dirPath, dirNames, fileNames in os.walk(SourcePath):
        for file in fileNames:
            if file[-4:] == ".bin":
                Type = file[11:13]
                year = file[14:18]
                month = file[18:20]
                if file[:7] == "wissdom":
                    Type = "WD"
                    year = file[26:30]
                    month = file[30:32]
                # if year != "2021":
                #     continue
                # 檢查檔案是否已存在
                if Type != "DZ" and os.path.exists(StorePath + Type + "/" + file[:-4] + ".npy"):
                    continue
                if Type == "DZ":
                    if dBZtoZ and os.path.exists(StorePath + "Z/" + file[:-4] + ".npy"):
                        continue
                    if not dBZtoZ and os.path.exists(StorePath + "DZ/" + file[:-4] + ".npy"):
                        continue
                # 儲存.npy
                if Type == "WD":
                    # continue
                    np_save(StorePath + "WD/" + file[:-4] + ".npy", ProcessNpy(os.path.join(dirPath, file), "WD"))
                elif Type == "DZ":
                    doc = "Z" if dBZtoZ else "DZ"
                    np_save(StorePath + doc + "/" + file[:-4] + ".npy",
                            ProcessNpy(os.path.join(dirPath, file), Type, dBZtoZ))
                else:
                    np_save(StorePath + Type + "/" + file[:-4] + ".npy", ProcessNpy(os.path.join(dirPath, file), Type))
            if file[-4:] == ".txt":
                # 檢查檔案是否已存在
                if os.path.exists(StorePath + "qpepre/" + file[:-4] + ".npy"):
                    continue
                # 儲存.npy
                try:
                    np_save(StorePath + "qpepre/" + file[:-4] + ".npy",
                            ProcessNpy(os.path.join(dirPath, file), "qpepre"))
                except Exception as e:
                    print(errMsg(e))

    # Generate_Hour_qpepre()
    print("ProcessData final")


def oldProcessData(dBZtoZ=True):
    def np_save(file, data):
        arr, msg = data
        if arr is None:
            print(msg)
        else:
            np.save(file, arr)

    def ProcessNpy(sourPath, Type, dBZtoZ=True):
        # 處理npy 和歸一化
        def ImgToWD(img_u, img_v, img_w):
            h, w = Get_LongLat(Region, True)
            img_u = cv2.resize(img_u, (w, h))
            img_v = cv2.resize(img_v, (w, h))
            img_w = cv2.resize(img_w, (w, h))
            img_u = np.where(img_u >= 0, np.log(img_u + 1) / np.log(MaxWD_u + 1),
                             -np.log(-img_u + 1) / np.log(-MinWD_u + 1))
            img_v = np.where(img_v >= 0, np.log(img_v + 1) / np.log(MaxWD_v + 1),
                             -np.log(-img_v + 1) / np.log(-MinWD_v + 1))
            img_w = np.where(img_w >= 0, np.log(img_w + 1) / np.log(MaxWD_w + 1),
                             -np.log(-img_w + 1) / np.log(-MinWD_w + 1))

            img = np.zeros((h, w, 48))
            for i in range(h):
                for j in range(w):
                    for k in range(8):
                        if img_u[i][j][k] >= 0:
                            img[i][j][k * 6 + 0] = img_u[i][j][k]
                        else:
                            img[i][j][k * 6 + 1] = -img_u[i][j][k]
                        if img_v[i][j][k] >= 0:
                            img[i][j][k * 6 + 2] = img_v[i][j][k]
                        else:
                            img[i][j][k * 6 + 3] = -img_v[i][j][k]
                        if img_w[i][j][k] >= 0:
                            img[i][j][k * 6 + 4] = img_w[i][j][k]
                        else:
                            img[i][j][k * 6 + 5] = -img_w[i][j][k]
            return img

        def Preprocess(array, Type):
            if Type == "KD":
                array = np.where(array > MaxKD, MaxKD, array)

            # array = cv2.resize(array, (192, 256))

            # dBZ to Z
            if Type == "Z":
                None
                # array = 10 ** (array / 10)
                # array = np.log(array + 1 - MinZ) / np.log(MaxZ - MinZ + 1)
            elif Type == "DR":  # DR
                array = np.log2(array + 1) / np.log2(MaxDR + 1)
            elif Type == "DZ":  # DZ
                array = np.log(array + 1 - MinDZ) / np.log(MaxDZ - MinDZ + 1)
            elif Type == "KD":  # KD
                array = np.log10(array + 1) / np.log10(MaxKD + 1)
            return array

        try:
            if Type == "WD":
                (img_u, img_v, img_w) = SourceToNpy(sourPath, Type, Split=True)
                return ImgToWD(img_u, img_v, img_w), ""
            elif Type in ["DR", "DZ", "KD"]:
                # dBZ to Z
                array = SourceToNpy(sourPath, Type, Split=True)
                if Type == "DZ" and dBZtoZ:
                    return Preprocess(array, "Z"), ""
                return Preprocess(array, Type), ""
            elif Type == "qpepre":
                return SourceToNpy(sourPath, Type, Split=True), ""
        except Exception as e:
            return None, errMsg(e)

    if not os.path.exists(Path_shp):
        raise Exception("please upload the shp file at ./shp/TWN_CITY.shp")

    def makedirs(Path_QPE):
        if not os.path.exists(Path_QPE):
            os.makedirs(Path_QPE)
        if not os.path.exists(Path_QPE + "/DR"):
            os.makedirs(Path_QPE + "/DR")
        # if not os.path.exists(Path_QPE + "/DZ"):
        #     os.makedirs(Path_QPE + "/DZ")
        if not os.path.exists(Path_QPE + "/KD"):
            os.makedirs(Path_QPE + "/KD")
        if not os.path.exists(Path_QPE + "/WD"):
            os.makedirs(Path_QPE + "/WD")
        if not os.path.exists(Path_QPE + "/qpepre"):
            os.makedirs(Path_QPE + "/qpepre")
        if not os.path.exists(Path_QPE + "/Z"):
            os.makedirs(Path_QPE + "/Z")

    makedirs(Path_QPE)
    StorePath = Path_QPE + "/"

    for dirPath, dirNames, fileNames in os.walk(SourcePath):
        for file in fileNames:
            if file[-4:] == ".bin":
                Type = file[11:13]
                if file[:7] == "wissdom":
                    Type = "WD"
                # 檢查檔案是否已存在
                if Type != "DZ" and os.path.exists(StorePath + Type + "/" + file[:-4] + ".npy"):
                    continue
                if Type == "DZ":
                    if dBZtoZ and os.path.exists(StorePath + "Z/" + file[:-4] + ".npy"):
                        continue
                    if not dBZtoZ and os.path.exists(StorePath + "DZ/" + file[:-4] + ".npy"):
                        continue
                # 儲存.npy
                if Type == "WD":
                    np_save(StorePath + "WD/" + file[:-4] + ".npy", ProcessNpy(os.path.join(dirPath, file), "WD"))
                elif Type == "DZ":
                    doc = "Z" if dBZtoZ else "DZ"
                    np_save(StorePath + doc + "/" + file[:-4] + ".npy",
                            ProcessNpy(os.path.join(dirPath, file), Type, dBZtoZ))
                else:
                    np_save(StorePath + Type + "/" + file[:-4] + ".npy", ProcessNpy(os.path.join(dirPath, file), Type))
            if file[-4:] == ".txt":
                # 檢查檔案是否已存在
                if os.path.exists(StorePath + "qpepre/" + file[:-4] + ".npy"):
                    continue
                # 儲存.npy
                try:
                    np_save(StorePath + "qpepre/" + file[:-4] + ".npy",
                            ProcessNpy(os.path.join(dirPath, file), "qpepre"))
                except Exception as e:
                    print(errMsg(e))

    # Generate_Hour_qpepre()
    print("ProcessData final")


def Generate_Hour_qpepre():
    preList = os.listdir(Path_QPE + r"/qpepre")
    for npy_fname in preList:
        start, end = npy_fname[7:19], npy_fname[20:32]
        start_d, end_d = datetime.datetime.strptime(start, "%Y%m%d%H%M"), \
                         datetime.datetime.strptime(end, "%Y%m%d%H%M")
        next = datetime.datetime.strftime(end_d + datetime.timedelta(hours=1), "%Y%m%d%H%M")
        next_npy_fname = "qpepre_" + end + "-" + next + "_1_h.npy"
        if next_npy_fname in preList:
            npy = np.load(os.path.join(Path_QPE, "qpepre", npy_fname))
            next_npy = np.load(os.path.join(Path_QPE, "qpepre", next_npy_fname))
            # for mins in [10, 20, 30, 40, 50]:
            for mins in [30]:
                be, af = datetime.datetime.strftime(start_d + datetime.timedelta(minutes=mins), "%Y%m%d%H%M"), \
                         datetime.datetime.strftime(end_d + datetime.timedelta(minutes=mins), "%Y%m%d%H%M")
                if not os.path.exists(os.path.join(Path_QPE, "qpepre", "qpepre_" + be + "-" + af + "_1_h.npy")):
                    array = (npy * (60 - mins) / 60) + (next_npy * mins / 60)
                    np.save(os.path.join(Path_QPE, "qpepre", "qpepre_" + be + "-" + af + "_1_h.npy"), array)


def copyData(min="30"):
    import shutil
    hourlist = ["20200505", "20200515", "20200525"
        , "20210505", "20210515", "20210525"
        , "20210605", "20210615", "20210625"
        , "20210705", "20210715", "20210725"]
    for dirPath, dirNames, fileNames in os.walk(r"D:/Data2"):
        for file in fileNames:
            if file[-4:] == ".bin":
                time = file[-16:-4]
                if time[-2:] != min:
                    continue
            elif file[-4:] == ".txt":
                time = file[7:19]
            else:
                continue
            for t in hourlist:
                if time[:len(t)] == t:
                    print(file)
                    shutil.copyfile(os.path.join(dirPath, file), os.path.join(SourcePath, file))


def Generate_site(source_path=r'./site.txt', save_path=r'./site.npy'):
    # 把測站座標點轉為.npy
    E = []
    N = []
    with open(source_path, "r") as f:
        line = f.readline()
        while line:
            E.append(float(line[8:16]))
            N.append(float(line[17:24]))
            line = f.readline()

    site = np.zeros((275, 162))
    for i in range(len(E)):
        try:
            site[round((N[i] - 21.8875) / 0.0125)][round((E[i] - 120) / 0.0125)] = 1
        except:
            print(N[i], E[i])
    site = site[::-1, :]

    np.save(save_path, site)


def errMsg(e):
    error_class = e.__class__.__name__  # 取得錯誤類型
    detail = e.args[0]  # 取得詳細內容
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
    fileName = lastCallStack[0]  # 取得發生的檔案名稱
    lineNum = lastCallStack[1]  # 取得發生的行號
    funcName = lastCallStack[2]  # 取得發生的函數名稱
    errMsg = "File \"{}\", line {}, in {}: [{}] {}, ".format(fileName, lineNum, funcName, error_class, detail)
    return errMsg


def CheckDataSet():
    # 檢查是否有遺漏資料
    qpepre, DR, DZ, KD, WD = Path_QPE + r"/qpepre", Path_QPE + r"/DR", Path_QPE + r"/DZ", Path_QPE + r"/KD", Path_QPE + r"/WD"
    preList, DRList, DZList, KDList, WDList = os.listdir(qpepre), os.listdir(DR), os.listdir(DZ), os.listdir(
        KD), os.listdir(WD)
    if len(DZList) == 0:
        DZ = Path_QPE + r"/Z"
        DZList = os.listdir(Path_QPE + r"/Z")

    start, end = datetime.datetime.strptime("2021/05/01 00:00", "%Y/%m/%d %H:%M"), \
                 datetime.datetime.strptime("2021/08/01 00:00", "%Y/%m/%d %H:%M")
    while start < end:
        b, e = datetime.datetime.strftime(start, "%Y%m%d%H%M"), \
               datetime.datetime.strftime(start + datetime.timedelta(hours=1), "%Y%m%d%H%M")

        dr, dz, kd, wd = BindName(b, ".npy")
        rain = "qpepre_" + b + "-" + e + "_1_h.npy"

        if dr in DRList and dz in DZList and kd in KDList and wd in WDList and rain in preList:
            start += datetime.timedelta(minutes=30)
            continue
        else:
            lack = b + " "
            if dr not in DRList:
                lack += dr + " "
            if dz not in DZList:
                lack += dz + " "
            if kd not in KDList:
                lack += kd + " "
            if wd not in WDList:
                lack += wd + " "
            if rain not in preList:
                lack += rain + " "

            print(lack)

        start += datetime.timedelta(minutes=30)


# 外部引用
def SourceToNpy(sourPath, Type, Split=False):
    def unpack(sourPath, Type):
        def DataToImg(data):
            c = str(501 * 381 * 21)
            img = np.array(struct.unpack(c + 'f', data)).reshape(21, 501, 381).transpose(1, 2, 0)[::-1, :, :]
            return img

        if Type == "WD":
            with open(sourPath, "rb") as f:
                data = f.read()
            img_u = DataToImg(data[6 * 381 * 501 * 21 * 4: 7 * 381 * 501 * 21 * 4])
            img_v = DataToImg(data[7 * 381 * 501 * 21 * 4: 8 * 381 * 501 * 21 * 4])
            img_w = DataToImg(data[8 * 381 * 501 * 21 * 4: 9 * 381 * 501 * 21 * 4])
            return img_u, img_v, img_w
        elif Type in ["DR", "DZ", "KD"]:
            with open(sourPath, "rb") as f:
                float_array = arr.array('f')
                float_array.fromfile(f, 441 * 561 * 34)
                array = np.array(float_array).reshape(-1).reshape(34, 561, 441).transpose(1, 2, 0)[::-1, :, :]
                return array
        elif Type == "qpepre":
            npy = []
            temp = 0
            with open(sourPath, "r") as f:
                lines = f.readline()
                while lines:
                    if lines[19:26] != temp:
                        npy.append([])
                        temp = lines[19:26]
                    npy[-1].append(float(lines[30:35]))
                    lines = f.readline()
            array = np.array(npy[::-1])
            return array

    # 將.bin, .txt 轉為npy, 並把-9999轉為最小值
    if sourPath is None:
        return None, "sourcePath is None" if Type != "WD" else None, None, None, "sourcePath is None"
    try:
        if Type == "WD":
            img_u, img_v, img_w = unpack(sourPath, Type)
            if Split:
                img_u = SplitRegion(img_u[:, :, :8], "WD", Region)
                img_v = SplitRegion(img_v[:, :, :8], "WD", Region)
                img_w = SplitRegion(img_w[:, :, 6:14], "WD", Region)
            img_u = np.where(img_u == -999, 0, img_u)
            img_v = np.where(img_v == -999, 0, img_v)
            img_w = np.where(img_w == -999, 0, img_w)

            return img_u, img_v, img_w
        elif Type in ["DR", "DZ", "KD"]:
            array = unpack(sourPath, Type)
            if Split:
                array = np.concatenate((array[:, :, 2:7], array[:, :, 28:31]), 2)
                array = SplitRegion(array, "Radar", Region)
            if Type == "DZ":
                array = np.where(array < MinDZ, MinDZ, array)
            else:
                array = np.where(array < 0, 0, array)
            return array
        elif Type == "qpepre":
            array = unpack(sourPath, Type)
            return array
    except Exception as e:
        raise Exception(sourPath, ": ", errMsg(e))


def Get_LongLat(Type, size=False):
    if size:  # get h, w
        if Type == "Radar" or Type == "DR" or Type == "DZ" or Type == "KD":
            return 561, 441
        if Type == "WD":
            return 501, 381
        if Type == "TW":
            return 275, 162
        if Type == "RS_TW":
            return 256, 192
        if Type == "NT":
            return 64, 64
        if Type == "ST":
            return 64, 64
    else:
        if Type == "Radar" or Type == "DR" or Type == "DZ" or Type == "KD":
            return 118, 123.5, 20, 27
        if Type == "WD":
            return 119, 122.8, 21, 26
        if Type == "TW":
            return 120, 122.0125, 21.8875, 25.3125
        if Type == "RS_TW":
            return 120, 122.0125, 21.8875, 25.3125
        if Type == "NT":
            return 122 - 0.0125 * 63, 122, 25.3 - 0.0125 * 63, 25.3
        if Type == "ST":
            return 120.1, 120.1 + 0.0125 * 63, 23.1, 23.1 + 0.0125 * 63

        if Type == "CAAH60":  # 大安森林
            return 121.5281, 25.0313
        if Type == "C0A950":  # 鼻頭角
            return 121.9153, 25.1308
        if Type == "C0A9F0":  # 內湖
            return 121.5672, 25.0812
        if Type == "C0U940":  # 羅東
            return 121.7411, 24.6836
        if Type == "C0C480":  # 桃園
            return 121.3150, 24.9943

        if Type == "467530":  # 阿里山
            return 120.8051, 23.5100
        if Type == "A0K420":  # 麥寮
            return 120.2096, 23.8005
        if Type == "C0X120":  # 麻豆
            return 120.2405, 23.1851
        if Type == "C0M530":  # 奮起湖
            return 120.6911, 23.4958
        if Type == "C0I380":  # 集集
            return 120.7933, 23.8300

    return None


def SplitRegion(img, be_region, af_region):
    if Get_LongLat(be_region) == Get_LongLat(af_region):
        return img
    be_long, be_long2, be_lat, be_lat2 = Get_LongLat(be_region)
    af_long, af_long2, af_lat, af_lat2 = Get_LongLat(af_region)

    assert be_lat2 >= af_lat2
    assert af_long >= be_long

    assert af_lat >= be_lat
    assert be_long2 >= af_long2

    if be_region == "WD":
        img = img[round((be_lat2 - af_lat2) / 0.01):round((be_lat - af_lat) / 0.01),
              round((af_long - be_long) / 0.01):round((af_long2 - be_long2) / 0.01)]
        return img
    img = img[round((be_lat2 - af_lat2) / 0.0125):round((be_lat - af_lat) / 0.0125),
          round((af_long - be_long) / 0.0125):round((af_long2 - be_long2) / 0.0125)]
    return img


def TimeToSourcePath(h, Type):
    dr, dz, kd, wd = BindName(h, ".bin")
    ob = dr if Type == "DR" else dz if Type == "DZ" else kd if Type == "KD" else wd if Type == "WD" else ""
    for dirPath, dirNames, fileNames in os.walk(SourcePath):
        for file in fileNames:
            if file == ob:
                return os.path.join(dirPath, file)
    return None


def BindName(h, end=".bin"):
    dr, dz, kd = "CAPPI_COMP_DR_" + h + end, "CAPPI_COMP_DZ_" + h + end, "CAPPI_COMP_KD_" + h + end
    wd = "wissdom_out_Taiwan_mosaic_" + h + end
    return dr, dz, kd, wd


if __name__ == "__main__":
    # copyData()
    ProcessData()
    Generate_Hour_qpepre()
    CheckDataSet()
    # Pearson_Correlation()
    # box_Data("WD")
    # box_Data("DR")
    # box_Data("KD")
    # box_Data("DZ")

    # ViewTest([
# "0215_2237",
# "0216_1126",])
