##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for meta-train phase. """
import os


def run_exp(seed=377, max_epoch=20, num_batch=1000, way=3, test_way=3, shot=1, query=15, lr1=0.0001, lr2=0.001, base_lr=0.01, update_step=10,
            gamma=0.5):
    # max_epoch = 20

    step_size = 10
    gpu = 0

    the_command = 'python main.py' \
                  + ' --seed=' + str(seed) \
                  + ' --max_epoch=' + str(max_epoch) \
                  + ' --num_batch=' + str(num_batch) \
                  + ' --shot=' + str(shot) \
                  + ' --train_query=' + str(query) \
                  + ' --way=' + str(way) \
                  + ' --test_way=' + str(test_way) \
                  + ' --meta_lr1=' + str(lr1) \
                  + ' --meta_lr2=' + str(lr2) \
                  + ' --step_size=' + str(step_size) \
                  + ' --gamma=' + str(gamma) \
                  + ' --gpu=' + str(gpu) \
                  + ' --base_lr=' + str(base_lr) \
                  + ' --update_step=' + str(update_step)

    os.system(the_command + ' --phase=meta_train')



# run_exp(seed=999, max_epoch=50, num_batch=30, way=3, test_way=3, shot=1, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #82.47 35  no gan 79.00
# run_exp(seed=999, max_epoch=50, num_batch=30, way=3, test_way=3, shot=5, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #88.76 40
# run_exp(seed=999, max_epoch=50, num_batch=30, way=3, test_way=3, shot=10, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #90.69 11

# run_exp(seed=999, max_epoch=50, num_batch=30, way=5, test_way=5, shot=1, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #71.31 24
# run_exp(seed=999, max_epoch=50, num_batch=30, way=5, test_way=5, shot=5, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)   #80.84 24
# run_exp(seed=999, max_epoch=50, num_batch=30, way=5, test_way=5, shot=10, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #83.48 35


run_exp(seed=999, max_epoch=50, num_batch=30, way=3, test_way=5, shot=1, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5) #71.29 46  69.22 24
# run_exp(seed=999, max_epoch=50, num_batch=30, way=3, test_way=5, shot=5, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5) #80.40
# run_exp(seed=999, max_epoch=50, num_batch=30, way=3, test_way=5, shot=10, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5) #83.15

# run_exp(seed=999, max_epoch=50, num_batch=30, way=5, test_way=3, shot=1, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #82.06 49    78.96
# run_exp(seed=999, max_epoch=50, num_batch=30, way=5, test_way=3, shot=5, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #88.50  39
# run_exp(seed=999, max_epoch=50, num_batch=30, way=5, test_way=3, shot=10, query=15, lr1=0.0001, lr2=0.0001, base_lr=0.05, update_step=200, gamma=0.5)  #90.41 39
















