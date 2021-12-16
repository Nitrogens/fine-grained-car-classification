import os
import sys
import argparse

import scipy.io as scio
import cv2

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/home/xmj/ml-course/car/dataset/cars_test_annos.mat")
parser.add_argument('--data_root', type=str, default="/home/xmj/ml-course/car/dataset/cars_test")
parser.add_argument('--label_path', type=str, default="/home/xmj/ml-course/car/dataset/test.label")
parser.add_argument('--save_path', type=str, default="/home/xmj/ml-course/car/dataset/cars_test_crop")
opts = parser.parse_args()

data = scio.loadmat(opts.data_path)

if not os.path.exists(opts.save_path):
    os.makedirs(opts.save_path)

with open(opts.label_path, "w") as f:
    # f.write(f"name\n")
    for data_line in tqdm(data["annotations"][0], "Data"):
        left_top = (int(data_line[0][0][0]), int(data_line[1][0][0]))
        right_bottom = (int(data_line[2][0][0]), int(data_line[3][0][0]))
        file_name = str(data_line[4][0])
        # print(left_top, right_bottom, label, file_name)
        img = cv2.imread(os.path.join(opts.data_root, file_name))
        img = img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0], :]
        cv2.imwrite(os.path.join(opts.save_path, file_name), img)
        f.write(f"{os.path.join(opts.save_path.split('/')[-1], file_name)}\n")