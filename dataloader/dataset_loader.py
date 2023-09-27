##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        #Set the path according to train, val and test
        if setname == 'pretrain':
            THE_PATH = osp.join(args.dataset_dir, 'pretrain')
            label_list = os.listdir(THE_PATH)  # 目录名即是标签
            # print(len(label_list))
        elif setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train')
            label_list = os.listdir(THE_PATH)  #目录名即是标签
            # print(len(label_list))
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'test')
            label_list = os.listdir(THE_PATH)
            # print(len(label_list))
        elif setname=='val':
            THE_PATH = osp.join(args.dataset_dir, 'val')
            label_list = os.listdir(THE_PATH)
            # print(len(label_list))
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]  #如果标签目录存在，则获得目录所在的路径

        # Get the images' paths and labels  获取标签目录内的图像数据地址并打上对应标签0-N
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))


        # Transformation 图片的格式转换，数据增强
        if train_aug:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),   #图片为84*84，这里resize成92*92，输入只有一个数时，是默认将短边缩放，然后另一个边按等比例缩放
                transforms.RandomResizedCrop(88),  #即是将给定图片随机增大或者缩小，然后按照88*88的大小进行裁剪
                transforms.CenterCrop(image_size), #中心裁剪，将图片裁剪成80*80
                transforms.RandomHorizontalFlip(), #依据概率p对PIL图片进行水平翻转，p默认0.5
                transforms.ToTensor(),
                #为何取这个mean和std原因未明
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                #为何取这个均值和方差，原因未名
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
