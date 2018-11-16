# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 12:02
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    f = open("./temp_data/npy.txt")
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        print(line)
        f, axs = plt.subplots(2, 1, figsize=(15, 5))
        axs[0].plot(np.load(line)[0:24 * 7])
        axs[1].plot(np.load(line)[0:24 * 7])
        axs[1].plot(np.load(line)[24 * 7 * 1:24 * 7 * 2])
        axs[1].plot(np.load(line)[24 * 7 * 2:24 * 7 * 3])
        axs[1].plot(np.load(line)[24 * 7 * 3:24 * 7 * 4])
        plt.xlabel("hour")
        plt.ylabel("number of in/out")
        if not os.path.exists("../data/first_month_figures"):
            os.makedirs("../data/first_month_figures")
        plt.savefig("../data/first_month_figures/%d.png" % (int(os.path.basename(line).split('.')[0])))
        plt.show()


if __name__ == "__main__":
    main()
