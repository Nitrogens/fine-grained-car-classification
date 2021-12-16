import model
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
  cur = 0
  timebegin = time.time()
  with open(os.path.join(savepath, "train_log"), 'w') as outfile:
    for epoch in range(1, config["params"]["epoch"]+1):
      for i, (data, label) in enumerate(dataset):
        # Acquire data
        data = data.to(device)
        label = label.to(device)
 
        # forward
        pred = net(data)

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
          log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
          print(log)
          outfile.write(log + "\n")
          sys.stdout.flush()   
          outfile.flush()

      scheduler.step()

      if epoch % config["save"]["step"] == 0:
        torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

