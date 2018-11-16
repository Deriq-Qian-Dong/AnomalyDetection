# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 9:14
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from config import *

Data = pd.read_csv('./temp_data/data.csv')
Indices = np.arange(Data.shape[0])  # 随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)
Data = np.array(Data)


def get_batch():
    global BATCH_START, BATCH_SIZE
    # xs shape (50batch, 20steps)
    data = Data[Indices[BATCH_START:BATCH_START + BATCH_SIZE]]
    seq = np.array([d[0:TIME_STEPS].tolist() for d in data])
    res = np.array([d[-1] for d in data]).reshape(-1, 1)
    BATCH_START += BATCH_SIZE
    return [seq[:, :, np.newaxis], res]


def get_data(num):
    data = Data[num]
    seq = np.array(data[0:TIME_STEPS])
    res = data[-1]
    return [seq.reshape(-1, 1), res]


def predict_by_model(seq):  # seq shape为 [TIME_STEPS, 1]  [24*7, 1]
    seq = np.array(seq)
    # reshape为[BATCH_SIZE, 24*7, 1]
    seq = np.array([seq.tolist() for _ in range(BATCH_SIZE)]).reshape(BATCH_SIZE, 24*7, 1)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
        graph = tf.get_default_graph()
        predict = graph.get_tensor_by_name("out_hidden/Wx_plus_b/Maximum:0")
        xs = graph.get_tensor_by_name("inputs/xs:0")
        cell_init_state = graph.get_tensor_by_name("LSTM_cell/initial_state/BasicLSTMCellZeroState/zeros:0")
        state = np.load("./model/final_state.npy")
        pred = sess.run(predict, feed_dict={xs: seq, cell_init_state: state})
    return pred[0][0]


