# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 12:21
@author: Eric
@email: qian.dong.2018@gmail.com
"""
from _5predict_by_model import predict_by_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def predict_5th_week(npy):
    test = np.load(npy)
    predicts = []
    for i in range(24 * 7):
        print(i, npy)
        seq = test[24 * 7 * 3 + i:24 * 7 * 4 + i]
        predicts.append(predict_by_model(seq))
    return predicts


def draw_predict(predicts, npy):
    test = np.load(npy)
    f, axs = plt.subplots(2, 1, figsize=(15, 5))
    axs[0].plot(test[24 * 7 * 4:24 * 7 * 5] - 3, 'r')
    axs[0].plot(test[24 * 7 * 4:24 * 7 * 5], 'b')
    axs[0].plot(test[24 * 7 * 4:24 * 7 * 5] + 3, 'r')
    axs[0].plot(predicts, 'g')
    axs[1].plot(test[24 * 7 * 4:24 * 7 * 5], 'b')
    axs[1].plot(predicts, 'g')
    plt.xlabel("hour")
    plt.ylabel("number of in/out")
    plt.savefig("../data/predict_result/%d.png"%(int(os.path.basename(npy).split('.')[0])))
    plt.show()


def main():
    f = open("./temp_data/npy.txt")
    lines = f.readlines()
    npys = [line.strip() for line in lines]
    for npy in npys[0:10]:
        predict = predict_5th_week(npy)
        draw_predict(predict, npy)


if __name__ == '__main__':
    main()
