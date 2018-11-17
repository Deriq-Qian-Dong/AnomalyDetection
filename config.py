# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 9:18
@author: Eric
@email: qian.dong.2018@gmail.com
"""
BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 168     # 时序数据的size
BATCH_SIZE = 256
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 64      # RNN 的 hidden unit size
LR = 1e-5           # learning rate