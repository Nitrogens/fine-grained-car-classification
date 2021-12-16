import model_moco as model
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
import os
import copy
import yaml
import random

def update_ema_params(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
  config = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)
  gpu_id = int(sys.argv[2])
  readername = config["reader"]
  dataloader = importlib.import_module("reader." + readername)
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

  print("Model building")
  net = model.model()
  net.train()
  net.to(device)
  net_ema = model.model()
  net_ema.train()
  net_ema.to(device)
  for param in net_ema.parameters():
    param.requires_grad_(False)

  print("optimizer building")
  lossfunc = config["params"]["loss"]
  loss_op = getattr(nn, lossfunc)().cuda()
  base_lr = config["params"]["lr"]

  decaysteps = config["params"]["decay_step"]
  decayratio = config["params"]["decay"]

  optimizer = optim.SGD(net.parameters(), lr=base_lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

  print("Training")
  length = len(dataset)
  total = length * config["params"]["epoch"]
  channel = 256
  queue = F.normalize(torch.randn(config["params"]["queue_len"], channel), dim=1).to(device)
  cur = 0
  timebegin = time.time()
  with open(os.path.join(savepath, "train_log"), 'w') as outfile:
    for epoch in range(1, config["params"]["epoch"]+1):
      for i, (data_0, data_1, label) in enumerate(dataset):
        # Acquire data
        data_0 = data_0.to(device)
        data_1 = data_1.to(device)
        label = label.to(device)

        le = data_0.shape[0]
 
        # forward
        pred_0, feature_0 = net(data_0)
        pred_1, feature_1 = net_ema(data_1)

        q = F.normalize(feature_0, dim=1)
        k = F.normalize(feature_1, dim=1)
        pos = torch.bmm(q.view(le, 1, channel), k.view(le, channel, 1))
        neg = torch.mm(q.view(le, channel), queue.t())
        logits = torch.cat((pos.view(le, 1), neg), dim=1)
        contras_loss = loss_op(logits / config["params"]["temp"], torch.zeros(le, dtype=torch.long).to(device))
        classify_loss = (loss_op(pred_0, label) + loss_op(pred_1, label)) / 2.

        # loss calculation
        loss = classify_loss + config["params"]["lambda_con"] * contras_loss
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()
        cur += 1

        update_ema_params(net, net_ema, 0.99, epoch * length + i)
        queue = torch.cat((queue[le:], F.normalize(feature_1, dim=1)), dim=0)

        # print logs
        if i % 20 == 0:
          timeend = time.time()
          resttime = (timeend - timebegin)/cur * (total-cur)/3600
          log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} classify_loss:{classify_loss} contras_loss:{contras_loss} lr:{base_lr}, rest time:{resttime:.2f}h"
          print(log)
          outfile.write(log + "\n")
          sys.stdout.flush()   
          outfile.flush()

      scheduler.step()

      if epoch % config["save"]["step"] == 0:
        torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

