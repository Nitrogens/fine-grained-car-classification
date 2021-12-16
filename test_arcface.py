import model_arcface as model
import importlib
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
  gpu_id = int(sys.argv[2])
  readername = config["reader_test"]
  dataloader = importlib.import_module("reader." + readername)

  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])
  device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
  
 
  savepath = os.path.join(loadpath, f"checkpoint")
  
  if not os.path.exists(os.path.join(loadpath, f"evaluation")):
    os.makedirs(os.path.join(loadpath, f"evaluation"))

  print("Read data")
  dataset = dataloader.txtload(labelpath, imagepath, 32, num_workers=4)

  begin = config["load"]["begin_step"]
  end = config["load"]["end_step"]
  step = config["load"]["steps"]

  for saveiter in range(begin, end+step, step):
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
    with torch.no_grad():
      with open(os.path.join(loadpath, f"evaluation/{saveiter}.log"), 'w') as outfile:
        for j, (data, name) in enumerate(dataset):
          data = data.to(device)
          preds = net(data, None, device, True)
          for k, pred in enumerate(preds):
            pred = pred.cpu().detach().numpy()
            pred_class = np.argmax(pred)
            count += 1
            outfile.write(f"{name[k]} {pred_class+1}\n")
            # accs += (pred_class == gts[k])
