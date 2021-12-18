import importlib
import numpy as np
import cv2 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import copy
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default="config/config_test.yaml")
  parser.add_argument('--load_path', type=str, default="/home/xmj/ml-course/car/experiments/baseline-model-model_resnet101-32-30-0.1-0.1-20-CrossEntropyLoss")
  parser.add_argument('--save_step', type=int, default=30)
  parser.add_argument('--model_name', type=str, default="model")
  parser.add_argument('--backbone', type=str, default="model_resnet101")
  parser.add_argument('--reader', type=str, default="reader_test")
  parser.add_argument('--i', type=int, default=-1)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=32)
  opts = parser.parse_args()

  config = yaml.load(open(opts.config), Loader = yaml.FullLoader)
  gpu_id = int(opts.gpu)
  readername = opts.reader
  dataloader = importlib.import_module("reader." + readername)
  model = importlib.import_module(opts.backbone)

  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = opts.model_name
  
  loadpath = os.path.join(opts.load_path)
  device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

  folder = os.listdir(labelpath)
  folder.sort()

  trains = copy.deepcopy(folder)
  if opts.i >= 0:
    vals = trains.pop(opts.i)
    print(f"Current Validation Set: {vals}")
  print(f"Current Training Set: {trains}")
  vallabelpath = os.path.join(labelpath, vals)
  
 
  savepath = os.path.join(loadpath, f"checkpoint/{opts.i}")
  
  if not os.path.exists(os.path.join(loadpath, f"evaluation")):
    os.makedirs(os.path.join(loadpath, f"evaluation"))

  print("Read data")
  dataset = dataloader.txtload(vallabelpath, imagepath, opts.batch_size, num_workers=4)

  with torch.no_grad():
    saveiter = opts.save_step
    print("Model building")
    net = model.model()
    statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"), map_location=device)

    net.to(device)
    net.load_state_dict(statedict)
    net.eval()

    print(f"Test {saveiter}")
    length = len(dataset)
    accs = 0
    count = 0
    
    with open("./submission.txt", 'w') as outfile:
      for j, (data, name) in enumerate(dataset):
        data = data.to(device)
        preds = net(data)
        for k, pred in enumerate(preds):
          pred = pred.cpu().detach().numpy()
          pred_class = np.argmax(pred)
          count += 1
          outfile.write(f"{name[k]} {pred_class+1}\n")
          # accs += (pred_class == gts[k])

