# -*- coding: utf-8 -*-
"""
Created on 2018/11/15 20:04
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import os
import numpy as np


def main():
    root_path = '../data/in_out/'
    save_list = []
    for root, dirs, files in os.walk(root_path):
        for npy in files:
            npy_ = os.path.join(root, npy)
            npy_ = np.load(npy_)
            pos_rate = 1 - sum(npy_ == 0) / len(npy_)  # 非0值的比例
            if pos_rate > 0.667:
                print(os.path.join(root, npy))
                save_list.append(os.path.join(root, npy))
    with open('npy.txt', 'w')as f:
        for name in save_list:
            f.write(name + "\n")


if __name__ == "__main__":
    main()
