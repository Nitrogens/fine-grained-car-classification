import os
import sys
import argparse
import random

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--train_list', type=str, default="/home/xmj/ml-course/car/dataset/train.label")
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--output_path', type=str, default="/home/xmj/ml-course/car/dataset/3-fold")
opts = parser.parse_args()

if not os.path.exists(opts.output_path):
    os.makedirs(opts.output_path)

with open(opts.train_list, "r") as f:
    file_data = f.readlines()
    random.shuffle(file_data)
    n = len(file_data)
    n_per_fold = n // opts.k
    for fold_id in tqdm(range(opts.k)):
        with open(os.path.join(opts.output_path, f"{fold_id}.label"), "w") as fw:
            if fold_id == opts.k - 1:
                for i in range(n_per_fold * fold_id, n):
                    fw.write(file_data[i])
            else:
                for i in range(n_per_fold * fold_id, n_per_fold * (fold_id + 1)):
                    fw.write(file_data[i])

