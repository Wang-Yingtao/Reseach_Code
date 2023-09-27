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
""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
# from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
from models.CenterLoss import CenterLoss


class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""

    def __init__(self, args):
        # Set the folder to save the records and checkpoints

        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
                     '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(
            args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
                     '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
                     '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
                     '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        # print(args.save_path)
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        self.train_way= self.args.way

        # Load meta-train set
        self.trainset = Dataset('train', self.args)
        # sampler的作用就是产生用于元学习的任务的意思
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way,
                                               self.args.shot + self.args.train_query)
        # sampler的作用就是产生一系列数据的index，而batch_sampler的作用就是产生一个batch的index
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=0,
                                       pin_memory=True)  # pin_memory的作用是拷贝到GPU中


        # Build meta-transfer learning model
        self.model = MtlLearner(self.args)

        # Set optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
             {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1, weight_decay=0.05)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                            gamma=self.args.gamma)

        # load pretrained model without FC classifier
        self.model_dict = self.model.state_dict()


        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
                args.pre_gamma) + '_step' + \
                             str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']

        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}

        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        if self.args.way != self.args.test_way:
            torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) + name + '.pth'))
        else:
            torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) + name + '.pth'))

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['train_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        torch.save(trlog, osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) + 'trlog'))

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
        # center_weights = self.ct_weights
        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            self.args.way = self.train_way
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Generate the labels for test set of the episodes during meta-train updates
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Using tqdm to read samples from train loader

            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                # Output logits for model
                logits_q, embedding_query, center_weights = self.model((data_shot, label_shot, data_query))

                # Calculate meta-train loss
                loss = F.cross_entropy(logits_q, label)

                # Calculate meta-train accuracy
                acc = count_acc(logits_q, label)

                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)


            # if self.args.way == self.args.test_way:
            #     self.save_model('model_epoch_acc')
            # else:
            #     self.save_model('encoder_epoch_acc')
            self.save_model('epoch_meta_train')

            if epoch > 10:
                self.eval(epoch)

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))




    def eval(self, epoch):
        """The function for the meta-eval phase."""
        # Load the logs
        trlog = torch.load(osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) + 'trlog'))

        if self.args.way == self.args.test_way:

            if self.args.eval_weights is not None:
                self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
            else:
                self.model.load_state_dict(torch.load(osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) +'epoch_meta_train' + '.pth'))['params'])

            # Load meta-test set
            test_set = Dataset('test', self.args)
            sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
            loader = DataLoader(test_set, batch_sampler=sampler, num_workers=0, pin_memory=True)

            # Set test accuracy recorder
            test_acc_record = np.zeros((600,))


            self.model.eval()

            # Set accuracy averager
            ave_acc = Averager()

            # Generate labels
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)

            # Start meta-test
            for i, batch in enumerate(loader, 1):
                if torch.cuda.is_available():
                        # print('len(batch)', len(batch[0]))
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                k = self.args.way * self.args.shot
                data_shot, data_query = data[:k], data[k:]
                    # print('data',len(data))
                    # print('data_query',len(data_query))
                logits_q, _, _ = self.model((data_shot, label_shot, data_query))
                acc = count_acc(logits_q, label)
                ave_acc.add(acc)
                test_acc_record[i - 1] = acc
                if i % 100 == 0:
                    print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

            # Calculate the confidence interval, update the logs
            m, pm = compute_confidence_interval(test_acc_record)
            print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

            if ave_acc.item() > trlog['max_acc']:
                trlog['max_acc'] = ave_acc.item()
                trlog['max_acc_epoch'] = epoch
                print('Best Epoch {}, Best Test Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
                torch.save(trlog, osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) + 'trlog'))

        else:
            self.args.way = self.args.test_way
            self.model_test = MtlLearner(self.args)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                # self.model = self.model.cuda()
                self.model_test = self.model_test.cuda()
            self.model_dict = self.model_test.state_dict()

            train_dict = torch.load(osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) +'epoch_meta_train.pth'))['params']

            train_dict = {'encoder.' + k: v for k, v in train_dict.items()}
            train_dict = {k: v for k, v in train_dict.items() if k in self.model_dict}
            # print(pretrained_dict.keys())
            self.model_dict.update(train_dict)
            self.model_test.load_state_dict(self.model_dict)

            # Load meta-test set
            test_set = Dataset('test', self.args)
            sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
            loader = DataLoader(test_set, batch_sampler=sampler, num_workers=0, pin_memory=True)

            # Set test accuracy recorder
            test_acc_record = np.zeros((600,))

            # Load model for meta-test phase
            # if self.args.eval_weights is not None:
            #     self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
            # else:
            #     self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
            # # Set model to eval mode
            self.model_test.eval()

            # Set accuracy averager
            ave_acc = Averager()

            # Generate labels
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)

            # Start meta-test
            for i, batch in enumerate(loader, 1):
                if torch.cuda.is_available():
                    # print('len(batch)', len(batch[0]))
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                k = self.args.way * self.args.shot
                data_shot, data_query = data[:k], data[k:]
                # print('data',len(data))
                # print('data_query',len(data_query))
                logits_q, _, _ = self.model_test((data_shot, label_shot, data_query))
                acc = count_acc(logits_q, label)
                ave_acc.add(acc)
                test_acc_record[i - 1] = acc
                if i % 100 == 0:
                    print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

            # Calculate the confidence interval, update the logs
            m, pm = compute_confidence_interval(test_acc_record)
            print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

            if ave_acc.item() > trlog['max_acc']:
                trlog['max_acc'] = ave_acc.item()
                trlog['max_acc_epoch'] = epoch

                torch.save(trlog, osp.join(self.args.save_path, str(self.train_way) + str(self.args.test_way) + 'trlog'))

            if epoch % 10 == 0:
                print('Best Epoch {}, Best Test Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))










