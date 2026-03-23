import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

def display(image):
    plt.imshow(image)
    plt.axis('off') 
    plt.show()

class Animefaces(Dataset):
    def __init__(self, data_folder):
        self.folder = os.path.join(data_folder, "images")
        self.files = [file for file in os.listdir(self.folder)]
        self.transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3) # [0, 1] -> [-1, 1] cause tanh in generator output 
            ]
        )

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.files[index])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    dataset = Animefaces("data")
    print(dataset[random.randint(0, 63000)].shape)



