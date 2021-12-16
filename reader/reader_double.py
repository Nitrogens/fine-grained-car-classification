import numpy as np
import cv2 
import os
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

transform_1 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
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

    image_path = os.path.join(self.root, line[0])
    img = Image.open(image_path)
    img_0 = transform(img)
    img_1 = transform_1(img)

    return img_0, img_1, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0):
  dataset = loader(labelpath, imagepath)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load

