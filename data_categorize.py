import os
import sys
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--train_list', type=str, default="/home/xmj/ml-course/car/dataset/train.label")
parser.add_argument('--output_path', type=str, default="/home/xmj/ml-course/car/dataset/list_per_class")
opts = parser.parse_args()

list_per_class = {}

with open(opts.train_list, "r") as f:
    file_data = f.readlines()
    for line_data in file_data:
        data_split = line_data.split(" ")
        file_name = data_split[0]
        label = int(data_split[1])
        if not label in list_per_class.keys():
            list_per_class[label] = []
        list_per_class[label].append(file_name)

for class_id in tqdm(range(196), "Class"):
    with open(os.path.join(opts.output_path, f"{class_id}.txt"), "w") as f:
        for file_name in list_per_class[class_id]:
            f.write(f"{file_name}\n")
