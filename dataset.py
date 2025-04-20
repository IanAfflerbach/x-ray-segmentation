import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CXRSegDataset(Dataset):
    def __init__(self):
        DATA_PATH = 'data/index.txt'

        self.ids = []
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r') as index_file:
                for line in index_file.readlines():
                    self.ids.append(line.replace('\n', ''))
        else:
            print("ERROR: index.txt not found")
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        label = self.get_mask(idx)
        tensor_img = transforms.ToTensor()(img)
        tensor_label = transforms.ToTensor()(label)
        return tensor_img, tensor_label
    
    def get_image(self, idx):
        id = self.ids[idx]
        return Image.open(f'data/xray-raw/{id}.jpg')
    
    def get_mask(self, idx):
        id = self.ids[idx]
        return Image.open(f'data/xray-segment-mask/{id}-mask.jpg')