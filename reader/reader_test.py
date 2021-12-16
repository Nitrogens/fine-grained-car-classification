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

    image_path = os.path.join(self.root, line[0])
    img = Image.open(image_path)
    img = transform(img)

    name = line[0].split("/")[-1]

    return img, name

def txtload(labelpath, imagepath, batch_size, shuffle=False, num_workers=0):
  dataset = loader(labelpath, imagepath)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load
