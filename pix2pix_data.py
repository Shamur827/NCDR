import numpy as np
import os
import cv2
import warnings
import struct
import array as arr

warnings.filterwarnings('ignore')

# SourcePath = r"./RadarData"
# Path_QPE = r"./QPE"
# Path_Pix2pix = r"./QPE/pix2pix"
# Path_shp = r'./RadarData/shp/TWN_CITY.shp'

SourcePath = r"E:\TWCC\RadarData"
SourcePath = r"D:/Data2"
Path_QPE = r"E:\TWCC/QPE"
Path_Pix2pix = r"E:\TWCC/QPE/pix2pix"
Path_shp = r"E:\TWCC/RadarData/shp/TWN_CITY.shp"
MaxDR, MinDR, MaxDZ, MinDZ, MaxKD, MinKD, MaxPre, MinPre, MaxZ, MinZ = \
    25.496891, 0.0, 110.35679, -30.603172, 39.522815, 0.0, 99.75, 0.0, 10 ** 11.035679, 10 ** -3.0603172
MaxWD_u, MinWD_u, MaxWD_v, MinWD_v, MaxWD_w, MinWD_w, = \
    26.52743215445662, -21.127624244138133, 30.38607470842544, -25.399568055872805, \
    11.52811917316285, -6.693125475125271


def ProcessData(dBZtoZ=True):
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
                # 檢查檔案是否已存在
                if file[:7] == "wissdom":
                    Type = "WD"
                if Type != "DZ" and os.path.exists(StorePath + Type + "/" + file[:-4] + ".npy"):
                    continue
                if Type == "DZ":
                    if dBZtoZ and os.path.exists(StorePath + "Z/" + file[:-4] + ".npy"):
                        continue
                    if not dBZtoZ and os.path.exists(StorePath + "DZ/" + file[:-4] + ".npy"):
                        continue
                #
                if Type == "WD":
                    np.save(StorePath + "WD/" + file[:-4] + ".npy", ProcessNpy(os.path.join(dirPath, file), "WD"))
                elif Type == "DZ":
                    doc = "Z" if dBZtoZ else "DZ"
                    np.save(StorePath + doc + "/" + file[:-4] + ".npy",
                            ProcessNpy(os.path.join(dirPath, file), Type, dBZtoZ))
                else:
                    np.save(StorePath + Type + "/" + file[:-4] + ".npy", ProcessNpy(os.path.join(dirPath, file), Type))
            if file[-4:] == ".txt":
                if os.path.exists(StorePath + "qpepre/" + file[:-4] + ".npy"):
                    continue
                np.save(StorePath + "qpepre/" + file[:-4] + ".npy", SourceToNpy(os.path.join(dirPath, file), ""))

    print("ProcessData final")


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


def BindName(h, end=".bin"):
    dr, dz, kd = "CAPPI_COMP_DR_" + h + end, "CAPPI_COMP_DZ_" + h + end, "CAPPI_COMP_KD_" + h + end
    wd = "wissdom_out_Taiwan_mosaic_" + h + end
    return dr, dz, kd, wd


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
            site[int((N[i] - 21.8875) // 0.0125)][int((E[i] - 120) // 0.0125)] = 1
        except:
            print(N[i], E[i])
    site = site[::-1, :]

    np.save(save_path, site)


def TimeToSourcePath(h, Type):
    dr, dz, kd, wd = BindName(h, ".bin")
    ob = dr if Type == "DR" else dz if Type == "DZ" else kd if Type == "KD" else wd if Type == "WD" else ""
    for dirPath, dirNames, fileNames in os.walk(SourcePath):
        for file in fileNames:
            if file == ob:
                return os.path.join(dirPath, file)
    return None


def SourceToNpy(sourPath, Type):
    # 將.bin, .txt 轉為npy, 並把-9999轉為最小值
    if sourPath is None:
        return None

    def DataToImg(data):
        c = str(501 * 381 * 21)
        img = np.array(struct.unpack(c + 'f', data)).reshape(21, 501, 381).transpose(1, 2, 0)[::-1, :, :]
        img = np.where(img == -999, 0, img)
        return img

    if sourPath[-4:] == ".bin":
        if Type == "WD":
            with open(sourPath, "rb") as f:
                data = f.read()
            img_u = DataToImg(data[6 * 381 * 501 * 21 * 4: 7 * 381 * 501 * 21 * 4])
            img_v = DataToImg(data[7 * 381 * 501 * 21 * 4: 8 * 381 * 501 * 21 * 4])
            img_w = DataToImg(data[8 * 381 * 501 * 21 * 4: 9 * 381 * 501 * 21 * 4])

            return (img_u, img_v, img_w)
        else:
            with open(sourPath, "rb") as f:
                float_array = arr.array('f')
                float_array.fromfile(f, 441 * 561 * 34)
                array = np.array(float_array).reshape(-1).reshape(34, 561, 441).transpose(1, 2, 0)[::-1, :, :]
                if Type == "DZ":
                    array = np.where(array == -9999, MinDZ, array)
                else:
                    array = np.where(array == -9999, 0, array)
                return array
    if sourPath[-4:] == ".txt":
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


def ProcessNpy(sourPath, Type, dBZtoZ=True):
    # 處理npy 切割區域和歸一化
    if sourPath is None:
        return None

    def ImgToWD(img_u, img_v, img_w):
        img_u = img_u[69:411, 100:301, :8]
        img_v = img_v[69:411, 100:301, :8]
        img_w = img_w[69:411, 100:301, :8]

        img_u = cv2.resize(img_u, (192, 256))
        img_v = cv2.resize(img_v, (192, 256))
        img_w = cv2.resize(img_w, (192, 256))
        img_u = np.where(img_u >= 0, np.log(img_u + 1) / np.log(MaxWD_u + 1),
                         -np.log(-img_u + 1) / np.log(-MinWD_u + 1))
        img_v = np.where(img_v >= 0, np.log(img_v + 1) / np.log(MaxWD_v + 1),
                         -np.log(-img_v + 1) / np.log(-MinWD_v + 1))
        img_w = np.where(img_w >= 0, np.log(img_w + 1) / np.log(MaxWD_w + 1),
                         -np.log(-img_w + 1) / np.log(-MinWD_w + 1))

        img = np.zeros((256, 192, 48))
        for i in range(256):
            for j in range(192):
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
        w = 192
        h = 256
        array = array[135:-151, 2/0.0125:-119, :8]
        if Type == "KD":
            array = np.where(array > MaxKD, MaxKD, array)

        array = cv2.resize(array, (w, h))

        # dBZ to Z
        if Type == "Z":
            array = 10 ** (array / 10)
            array = np.log(array + 1 - MinZ) / np.log(MaxZ - MinZ + 1)

        if Type == "DR":  # DR
            array = np.log2(array + 1) / np.log2(MaxDR + 1)
        elif Type == "DZ":  # DZ
            array = np.log(array + 1 - MinDZ) / np.log(MaxDZ - MinDZ + 1)
        elif Type == "KD":  # KD
            array = np.log10(array + 1) / np.log10(MaxKD + 1)
        return array

    if Type == "WD":
        (img_u, img_v, img_w) = SourceToNpy(sourPath, Type)
        return ImgToWD(img_u, img_v, img_w)
    else:
        # dBZ to Z
        if Type == "DZ" and dBZtoZ:
            return Preprocess(SourceToNpy(sourPath, Type), "Z")
        return Preprocess(SourceToNpy(sourPath, Type), Type)


if __name__ == "__main__":
    # copyData()
    ProcessData()
