# -*- coding: utf-8 -*-
"""
Created on 2018/11/15 17:33
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from LSTM_model import LSTMRNN
import matplotlib.pyplot as plt
from config import *

Data = pd.read_csv('./temp_data/data.csv')
Indices = np.arange(Data.shape[0])
np.random.shuffle(Indices)
Data = np.array(Data)


def get_batch():
    global BATCH_START, BATCH_SIZE
    data = Data[Indices[BATCH_START:BATCH_START + BATCH_SIZE]]
    seq = np.array([d[0:TIME_STEPS].tolist() for d in data])
    res = np.array([d[-1] for d in data]).reshape(-1, 1)
    BATCH_START += BATCH_SIZE
    return [seq[:, :, np.newaxis], res]


if __name__ == '__main__':
    with tf.Session() as sess:
        model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        costs = []
        begin = True
        # 训练 10 次
        for epoch in range(100):
            BATCH_START = 0
            np.random.shuffle(Indices)
            for i in range(len(Indices) // BATCH_SIZE):
                seq, res = get_batch()  # 提取 batch data
                if begin:
                    begin = False
                    # 初始化 data
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                    }
                else:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.cell_init_state: state  # 保持 state 的连续性
                    }
                # 训练
                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                print('epoch:%d step:%d cost: ' % (epoch, i), round(cost, 4))
                costs.append(cost)
                # print(pred)
        np.save("./model/final_state.npy", state)
        np.save("./temp_data/cost.npy", costs)
        plt.plot(costs)
        plt.show()
        plt.savefig("./temp_data/cost.png")
        if not os.path.exists("model"):
            os.makedirs("model")
        saver.save(sess=sess, save_path=os.path.join(os.getcwd(), 'model', 'model.ckpt'))
