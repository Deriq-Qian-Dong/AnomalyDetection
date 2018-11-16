# -*- coding: utf-8 -*-
"""
Created on 2018/11/16 11:35
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import numpy as np
import pandas as pd


def main():
    f = open('./temp_data/npy.txt')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.close()
    data = np.array([[0] * 169] * 15063)
    cnt = 0
    for idx, line in enumerate(lines):
        npy = np.load(line)
        offset = idx % (24 * 7)
        for i in range((2088 - offset - 1) // (24 * 7)):
            data[cnt] = (npy[i * 24 * 7 + offset:(i + 1) * 24 * 7 + offset + 1])
            cnt += 1
    data = pd.DataFrame(data)
    data.to_csv('./temp_data/data.csv', index=False)


if __name__ == "__main__":
    main()
