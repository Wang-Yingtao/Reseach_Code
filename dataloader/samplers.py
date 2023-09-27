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
""" Sampler for dataloader. """
import torch
import numpy as np

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        #下面for循环的作用就是获取每个label所处的位置索引
        for i in range(max(label) + 1):
            # print('ind_before',np.argwhere(label == i))
            ind = np.argwhere(label == i).reshape(-1) #np.argwhere是返回label=i的索引，得出来的是(n,1),reshape(-1)后则变为(1,n) ,https://blog.csdn.net/a1059682127/article/details/88052321,https://blog.csdn.net/u012193416/article/details/79672514
            # print('ind',ind)
            ind = torch.from_numpy(ind)  #转换成tensor
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls] #torch.randperm打乱顺序，返回0到n-1的数组，再取前n_cls个数
            # print('classes', classes)
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            # print('1',batch)
            # print('2',torch.stack(batch))
            # print('3', torch.stack(batch).t())    #t()这里是转置
            # print('4', torch.stack(batch).t().reshape(-1))
            batch = torch.stack(batch).t().reshape(-1)

            yield batch
