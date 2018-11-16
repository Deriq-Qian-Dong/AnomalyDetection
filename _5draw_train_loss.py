# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 11:15
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt


def main():
    costs = np.load('cost.npy')
    plt.plot(costs)
    plt.xlabel("step")
    plt.ylabel("mean squre error")
    plt.title("loss-step graph")
    plt.show()
    costs = costs.reshape(10, -1)
    f, axs = plt.subplots(1, 1, figsize=(15, 5))
    axs.boxplot([cost for cost in costs])
    plt.ylabel("mean square error")
    plt.xlabel("epoch")
    plt.title("loss-epoch box graph")
    plt.show()


if __name__ == "__main__":
    main()
