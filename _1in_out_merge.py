# -*- coding: utf-8 -*-
"""
Created on 2018/11/15 20:04
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import os
import numpy as np


def main():
    if not os.path.exists("./in"):
        os.makedirs("./in")
    if not os.path.exists("./out"):
        os.makedirs("./out")
    gridID_in_hourly = {}
    gridID_out_hourly = {}
    for i in range(104000):
        gridID_in_hourly[i] = [0] * 2088
        gridID_out_hourly[i] = [0] * 2088
    root_path = "/datahouse/tripflow/ano_detect/200/bj-byhour-io/"
    for io_file in os.listdir(root_path):
        hour = int(io_file.split('-')[-1])
        io_file = os.path.join(root_path, io_file)
        print(io_file)
        with open(io_file) as f:
            lines = f.readlines()
            lines = [line.strip().split(',') for line in lines]
            lines = np.array(lines)
            lines = lines.astype(int)
            for line in lines:
                gridID_out_hourly[line[0]][hour] = line[1]
                gridID_in_hourly[line[0]][hour] = line[2]
    for ID in gridID_in_hourly:
        np.save("./in/%d.npy" % ID, gridID_in_hourly[ID])
        np.save("./out/%d.npy" % ID, gridID_out_hourly[ID])


if __name__ == "__main__":
    main()
