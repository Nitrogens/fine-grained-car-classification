# import model
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import argparse

from model_arcface import FocalLoss

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default="config/config_val.yaml")
  parser.add_argument('--save_path', type=str, default="/home/xmj/ml-course/car/experiments/arcface")
  parser.add_argument('--save_step', type=int, default=10)
  parser.add_argument('--model_name', type=str, default="model")
  parser.add_argument('--backbone', type=str, default="model_arcface")
  parser.add_argument('--reader', type=str, default="reader")
  parser.add_argument('--i', type=int, default=-1)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--epoch', type=int, default=50)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--decay', type=float, default=0.1)
  parser.add_argument('--decay_step', type=int, default=20)
  parser.add_argument('--loss', type=str, default="FocalLoss")
  parser.add_argument('--s', type=float, default=64.0)
  parser.add_argument('--m', type=float, default=0.50)
  parser.add_argument('--easy_margin', action='store_true')
  parser.add_argument('--alpha', type=float, default=0.5)
  parser.add_argument('--gamma', type=float, default=2.0)
  opts = parser.parse_args()

  opts.save_path += f"-{opts.model_name}-{opts.backbone}-{opts.batch_size}-{opts.epoch}-{opts.lr}-{opts.decay}-{opts.decay_step}-{opts.loss}-{opts.s}-{opts.m}-{opts.easy_margin}-{opts.alpha}-{opts.gamma}"

  config = yaml.load(open(opts.config), Loader=yaml.FullLoader)
  gpu_id = int(opts.gpu)
  readername = opts.reader
  dataloader = importlib.import_module("reader." + readername)
  model = importlib.import_module(opts.backbone)
  device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

  config = config["train"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = opts.model_name

  folder = os.listdir(labelpath)
  folder.sort()

  trains = copy.deepcopy(folder)
  if opts.i >= 0:
    vals = trains.pop(opts.i)
    print(f"Current Validation Set: {vals}")
  print(f"Current Training Set: {trains}")
  trainlabelpath = [os.path.join(labelpath, j) for j in trains] 
  
  savepath = os.path.join(opts.save_path, f"checkpoint/{opts.i}")
  if not os.path.exists(savepath):
    os.makedirs(savepath)

  print("Read data")
  dataset = dataloader.txtload(trainlabelpath, imagepath, opts.batch_size, shuffle=True, num_workers=4)

  print("Model building")
  net = model.model(s=opts.s, m=opts.m, easy_margin=opts.easy_margin)
  net.train()
  net.to(device)

  print("optimizer building")
  lossfunc = opts.loss
  loss_op = FocalLoss(alpha=opts.alpha, gamma=opts.gamma).to(device)
  base_lr = opts.lr

  decaysteps = opts.decay_step
  decayratio = opts.decay

  optimizer = optim.SGD(net.parameters(), lr=base_lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

  print("Training")
  length = len(dataset)
  total = length * opts.epoch
  cur = 0
  timebegin = time.time()
  with open(os.path.join(savepath, "train_log"), 'w') as outfile:
    for epoch in range(1, opts.epoch+1):
      for i, (data, label, _) in enumerate(dataset):
        # Acquire data
        data = data.to(device)
        label = label.to(device)
 
        # forward
        pred = net(data, label, device)

        # loss calculation
        loss = loss_op(pred, label)
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()
        cur += 1

        # print logs
        if i % 20 == 0:
          timeend = time.time()
          resttime = (timeend - timebegin)/cur * (total-cur)/3600
          log = f"[{epoch}/{opts.epoch}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
          print(log)
          outfile.write(log + "\n")
          sys.stdout.flush()   
          outfile.flush()

      scheduler.step()

      if epoch % opts.save_step == 0:
        torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

