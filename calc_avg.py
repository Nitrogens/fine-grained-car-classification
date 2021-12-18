import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val_file', type=str, default="/home/xmj/ml-course/car/experiments/baseline-model-model_resnet101-32-50-0.1-0.1-20-CrossEntropyLoss/evaluation/all.log")
parser.add_argument('--save_path', type=str, default="/home/xmj/ml-course/car/experiments/baseline-model-model_resnet101-32-50-0.1-0.1-20-CrossEntropyLoss/evaluation/all_avg.log")
opts = parser.parse_args()

epoch_data = {}
fold_data = set({})

with open(opts.val_file, "r") as f:
    data = f.readlines()
    for data_line in data:
        data_split = data_line.strip().split(" ")
        fold_id = int(data_split[0])
        fold_data.add(fold_id)
        epoch = int(data_split[1])
        acc = float(data_split[-1])
        if not epoch in epoch_data:
            epoch_data[epoch] = 0
        epoch_data[epoch] += acc
    for epoch in epoch_data.keys():
        epoch_data[epoch] /= (1. * len(fold_data))
best_acc = 0.
best_epoch = -1
with open(opts.save_path, "w") as f:
    for epoch in epoch_data.keys():
        f.write(f"[Epoch:{epoch}] Accuracy: {epoch_data[epoch]}\n")
        if epoch_data[epoch] > best_acc:
            best_acc = epoch_data[epoch]
            best_epoch = epoch
    f.write(f"[Best Epoch:{best_epoch}] Accuracy: {best_acc}\n")