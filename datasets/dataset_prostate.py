import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
import torch.utils.data as data
import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as TF
import torchvision
# import pydicom
import numpy as np
import re
import math
from PIL import Image
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms


def load_data(path, remove_blank=False, prostate=True, bowel=True, NVB=True, vesicle=True, DVC=True,):
    """
    path: the input should be the patient dir's path.
    remove_blank: to remove gt frame without any class in it
    prostate,bowel,NVB,vesicle,DVC: set False to filter out unwanted classes
    :return: slices(num, H, W), gts(num, H, W), all in type ndarray
    """
    lst = [prostate, bowel, NVB, vesicle, DVC]
    assert np.sum(lst) > 0  # 分类任务至少与背景有两个类

    # find out nii file's name
    path_1 = path + '/T2'
    files = os.listdir(path_1)
    for f in files:
        if re.match(r'.*\.nii', f):
            nii_name = f

    # load nii
    nii = nib.load(os.path.join(path_1, nii_name))
    gts = nii.get_data()
    gts = np.array(gts)  # change type to ndarray for uniformity
    gts = np.transpose(gts, (2, 1, 0))  # for unknown reason, H & W of nii are reversed.
    gts[gts > 0] -= 1  # to make labels start at 1

    gts = np.array(gts).astype(np.int32)  # loss func 要求label是int

    # deal with labels stored in lst
    for idx, ele in enumerate(lst):
        if not ele:
            gts[gts == idx+1] = 0  # change unwanted class to background.
    ulst = np.unique(gts)  # ulst is already sorted.
    for idx, ele in enumerate(ulst):
        if ele > 0:  # 0 stands for background and doesn't need handling
            gts[gts == ele] = idx

    # load slices
    slices = np.load(os.path.join(path, 'normed.npy'))
    tgts = []
    tslices = []
    # print(nii_name, 'gt:',gts.shape)
    for idx in range(gts.shape[0]):
        # 去除背景
        if remove_blank:
            if len(np.unique(gts[idx])) > 1:  # 如果所有的像素都是同一个类别说明是背景
                tgts.append(gts[idx])
                tslices.append(slices[idx])
        else:
            tgts.append(gts[idx])
            tslices.append(slices[idx])
    gts = np.array(tgts)
    slices = np.array(tslices)
    # print(nii_name,'img:', slices.shape)

    return slices, gts


class Prostate_dataset(data.Dataset):
    def __init__(self, patients_path, train=True, trans=True, convert_tensor=True,
                 remove_blank=False,
                 prostate=True, bowel=True, NVB=True, vesicle=True, DVC=True,
                 ):  # patients_path:/mnt/ST4T-1/lxj/datasets/cases60/train/
        assert os.path.exists(patients_path)  # 断言 如果path存在，返回True；如果path不存在，返回False
        patients = os.listdir(patients_path)  # 返回病人列表
        if '.DS_Store' in patients:
            patients.remove('.DS_Store')

        self.im = []  # 全体病人的slices合集
        self.gt = []
        self.trans = trans
        self.convert_tensor = convert_tensor
        self.train = train
        for patient in patients:
            ims, gts = load_data(os.path.join(patients_path, patient),
                                     remove_blank, prostate, bowel, NVB, vesicle, DVC)
            if self.train:
                for idx in range(ims.shape[0]):
                    self.im.append(ims[idx])
                    self.gt.append(gts[idx])
            else:
                self.im.append(ims)
                self.gt.append(gts)


    def transform(self, image, mask):
        # change type to PIL in order to use module TF

        # if self.train:
        #     image = Image.fromarray(image.astype(float))
        #     mask = Image.fromarray(mask)
        #     # Random horizontal flipping 随机水平翻转
        #     if np.random.random() > 0.7:
        #         image = torchvision.transforms.functional.hflip(image)
        #         mask = torchvision.transforms.functional.hflip(mask)
        #     # Random crop 随机裁剪
        #     i, j, h, w = transforms.RandomCrop.get_params(
        #         image, output_size=(224, 224)
        #     )
        #     image = torchvision.transforms.functional.crop(image, i, j, h, w)
        #     mask = torchvision.transforms.functional.crop(mask, i, j, h, w)

        #     # add noise to image
        #     image += np.floor(4 * np.random.rand()) * np.floor(np.random.randn(224, 224)).clip(min=0)
        #     image = np.array(image)
        if self.train:
            image = Image.fromarray(image.astype(float)) 
            mask = Image.fromarray(mask)
            image_1 = torchvision.transforms.functional.center_crop(image, output_size=(224, 224))
            mask_1 = torchvision.transforms.functional.center_crop(mask, output_size=(224, 224))
        else:
            image_1 = np.empty([24, 224, 224], dtype = float)
            mask_1 = np.empty([24, 224, 224], dtype = float)
            for i in range(0, image.shape[0]):
                # print(image.shape[0])
                img = Image.fromarray(image[i].astype(float))
                mas = Image.fromarray(mask[i])
                image_1[i] = torchvision.transforms.functional.center_crop(img, output_size=(224, 224))
                mask_1[i] = torchvision.transforms.functional.center_crop(mas, output_size=(224, 224))


        # normalize
        # 0.7469453080849382 18.174489548723503

        # to numpy array
        # image = np.array(image)
        image_1 = (image_1 - 0.7469) / 18.1744
        mask_1 = np.array(mask_1).astype(np.int32)
        return image_1, mask_1

    def __getitem__(self, index):
        im, gt = self.transform(self.im[index], self.gt[index])
        if self.train:
            if self.convert_tensor:
                im = torch.Tensor(im).reshape(1, 224, 224)
                gt = torch.Tensor(gt).reshape(224, 224)
        else:
            im = torch.Tensor(im)
            gt = torch.Tensor(gt)
        # print('index:', index, 'im:', im.shape, 'gt:', gt.shape)
        return im, gt, index

    def __len__(self):
        return len(self.im)


if __name__ == '__main__':
    def one_hot_encoder(n_classes, input_tensor):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    path = '/mnt/HDD2/xxy/data_all/train'
    train_loader = DataLoader(
        dataset=Prostate_dataset(path, train=False, DVC=False, NVB=False),
        batch_size=1,
    )
    length = len(train_loader)
    pa = []
    da = []
    background_pos = torch.zeros([24, 224, 224])
    prostate_pos = torch.zeros([24, 224, 224])
    bowel_pos = torch.zeros([24, 224, 224])
    vesicle_pos = torch.zeros([24, 224, 224])
    # zero_pos = torch.zeros([224,224])
    # for i in range(0, 4):
    #     pa.append(torch.zeros([224,224]))
    for i_batch, sampled_batch in enumerate(train_loader):
        # print(sampled_batch[0].shape, sampled_batch[1].shape)
        sampled_batch[1] = torch.squeeze(sampled_batch[1])
        # print(sampled_batch[1].shape)
        # sampled_batch[1] 24 * 224 * 224
        for i in range(0, sampled_batch[1].shape[0]):
            gt = one_hot_encoder(4, sampled_batch[1])
            # print(gt[i, 0, :, :].shape)
            background_pos[i, :, :] = background_pos[i, :, :] + gt[i, 0, :, :]
            prostate_pos[i, :, :] = prostate_pos[i, :, :] + gt[i, 1, :, :]
            bowel_pos[i, :, :] = bowel_pos[i, :, :] + gt[i, 2, :, :]
            vesicle_pos[i, :, :] = vesicle_pos[i, :, :] + gt[i, 3, :, :]
    background_pos = background_pos / length
    prostate_pos = prostate_pos / length
    bowel_pos = bowel_pos / length
    vesicle_pos = vesicle_pos / length
    img_pos = torch.cat((torch.unsqueeze(background_pos, dim=1), torch.unsqueeze(prostate_pos, dim=1)), dim = 1)
    img_pos = torch.cat((img_pos, torch.unsqueeze(bowel_pos, dim=1)), dim=1) 
    img_pos = torch.cat((img_pos, torch.unsqueeze(vesicle_pos, dim=1)), dim=1)

    for i in range(0,img_pos.shape[0]):
        for class_num in range(0, img_pos.shape[1]):
            img_pos[i, class_num,:,:] = torch.where(img_pos[i, class_num,:,:]>0.5, torch.exp(-(img_pos[i, class_num,:,:] - 0.5).pow(2)/(2*0.2*0.2)),torch.exp(-(img_pos[i, class_num,:,:] - 0.5).pow(2)/(2*0.8*0.8)))
            # if img_pos[i, class_num,:,:] >0.5:
            #     img_pos[i, class_num,:,:] = torch.exp(-(img_pos[i, class_num,:,:] - 0.5)/(2*0.4*0.4))
            # else:
            #     img_pos[i, class_num,:,:] = torch.exp(-(img_pos[i, class_num,:,:] - 0.5)/(2*0.6*0.6))

    # a = math.exp(length)
    # print(a)

    Pa_final = torch.zeros([24, 4, 224, 224])
    for i in range(0,img_pos.shape[0]):
        for j in range(0, img_pos.shape[1]):
            pa[i] = pa[i] / length
            Pa_final[i, :, :] = pa[i]
            max_num = torch.max(pa[i])
            min_num = torch.min(pa[i])
            pa[i] = ( pa[i] - min_num ) / (max_num - min_num)

    # root_path = '/mnt/HDD2/xxy/'
    # q = []
    # for i in range(0, 24):
    #     # print(da[0][i, :, :].shape)
    #     q.append(img_pos[i,1,:,:])
    #     q[i] = 255 * q[i].numpy()
    #     img = Image.fromarray(q[i])
    #     path = root_path + '%d.jpg' % i
    #     img.convert('L').save(path)

    
    # save_path = '/mnt/HDD2/xxy/aa.jpg'
    # img = 255 *  pa[3].numpy()
    # im = Image.fromarray(img)
    # im.convert('L').save(save_path) # 保存为灰度图(8-bit)
    # im.convert('RGB').save(save_path) # 保存为RGB图(24-bit)