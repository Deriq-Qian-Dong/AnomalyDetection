# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 9:18
@author: Eric
@email: qian.dong.2018@gmail.com
"""
BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 168     # backpropagation through time 的 time_steps
BATCH_SIZE = 256
INPUT_SIZE = 1      # sin 数据输入 size
OUTPUT_SIZE = 1     # cos 数据输出 size
CELL_SIZE = 64      # RNN 的 hidden unit size
LR = 1e-5           # learning rate