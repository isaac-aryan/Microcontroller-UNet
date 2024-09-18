import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import torch

# Defining the Dataflow & Dataset

class ImgDataset(Dataset):

    def __init__(self, root_path, test=False):

        self.root_path = root_path

        if test:
            self.images = sorted([root_path+"/test/images/"+i for i in os.listdir(root_path+"/test/images")])
            self.masks = sorted([root_path+"/test/masks/"+i for i in os.listdir(root_path+"/test/masks/")])
        
        else:
            self.images = sorted([root_path+"/train/images/"+i for i in os.listdir(root_path+"/train/images")])
            self.masks = sorted([root_path+"/train/masks/"+i for i in os.listdir(root_path+"/train/masks")])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()] # dataset as a tensor
        )

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        
        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask # returns the image and corresponding ground truth
    
    def __len__(self):

        return len(self.images)
    