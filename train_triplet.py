from __future__ import print_function
from __future__ import division

import model_moco as model
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import math

from model_arcface import FocalLoss

if __name__ == "__main__":
  config = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)
  gpu_id = int(sys.argv[2])
  readername = config["reader"]
  dataloader = importlib.import_module("reader." + readername)
  dataloader_n = importlib.import_module("reader.reader")
  device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

  config = config["train"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["save"]["model_name"]

  savepath = os.path.join(config["save"]["save_path"], f"checkpoint/")
  if not os.path.exists(savepath):
    os.makedirs(savepath)

  print("Read data")
  dataset = dataloader.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4)
  dataset_n = dataloader_n.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4)

  print("Model building")
  net = model.model()
  net.train()
  net.to(device)

  print("optimizer building")
  loss_op = FocalLoss()
  triplet_loss_op = nn.TripletMarginLoss(margin=config["params"]["margin"])
  base_lr = config["params"]["lr"]

  decaysteps = config["params"]["decay_step"]
  decayratio = config["params"]["decay"]

  optimizer = optim.SGD(net.parameters(), lr=base_lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

  print("Training")
  length = len(dataset)
  total = length * config["params"]["epoch"]
  cur = 0
  timebegin = time.time()
  with open(os.path.join(savepath, "train_log"), 'w') as outfile:
    for epoch in range(1, config["params"]["epoch"]+1):
      for i, data in enumerate(zip(dataset, dataset_n)):
        # Acquire data
        (data_a, data_p, label_a), (data_n, label_n) = data
        data_a = data_a.to(device)
        data_p = data_p.to(device)
        data_n = data_n.to(device)
        label_a = label_a.to(device)
        assert data_a.shape[1] == 3
        assert data_p.shape[1] == 3
        assert data_n.shape[1] == 3
        assert data_a.shape[2] == data_a.shape[2] == 448
        assert data_p.shape[2] == data_p.shape[2] == 448
        assert data_n.shape[2] == data_n.shape[2] == 448
 
        # forward
        pred_a, feature_a = net(data_a)
        pred_p, feature_p = net(data_p)
        pred_n, feature_n = net(data_n)

        # loss calculation
        loss_classify = loss_op(pred_a, label_a)
        loss_triplet = triplet_loss_op(feature_a, feature_p, feature_n)
        loss = loss_classify + config["params"]["lambda_triplet"] * loss_triplet
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()
        cur += 1

        # print logs
        if i % 20 == 0:
          timeend = time.time()
          resttime = (timeend - timebegin)/cur * (total-cur)/3600
          log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} loss_classify:{loss_classify} loss_triplet:{loss_triplet} lr:{base_lr}, rest time:{resttime:.2f}h"
          print(log)
          outfile.write(log + "\n")
          sys.stdout.flush()   
          outfile.flush()

      scheduler.step()

      if epoch % config["save"]["step"] == 0:
        torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

