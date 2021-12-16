import numpy as np
import cv2 
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

from PIL import Image

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    # transforms.RandomCrop((224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class loader(Dataset): 
  def __init__(self, path, root):
    self.lines = []
    with open(path) as f:
      self.lines = f.readlines()
    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    # print(line)
    label = int(line[1])

    with open(os.path.join(self.root, "..", "list_per_class", f"{label}.txt"), r) as fr:
      file_data = fr.strip().readlines()
      file_data.pop(line[0])
      sampled_data = random.sample(file_data, 1)[0]


    image_path = os.path.join(self.root, line[0])
    img = Image.open(image_path)
    img = transform(img)

    image_path_con = os.path.join(self.root, sampled_data)
    img_con = Image.open(image_path_con)
    img_con = transform(img_con)

    return img, img_con, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0):
  dataset = loader(labelpath, imagepath)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load
