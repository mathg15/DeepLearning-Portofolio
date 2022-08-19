import itertools

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, models, transforms
import time
import matplotlib.pyplot as plt
from torchvision.io import read_image

assert torch.cuda.is_available()
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

labels_map = {0: 'buildings',
              1: 'forest',
              2: 'glacier',
              3: 'mountain',
              4: 'sea',
              5: 'street'}


pred_path = "seg_pred/seg_pred"

pred_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 100))])
predset = datasets.ImageFolder(pred_path, transform=pred_transform)


pred_loader = DataLoader(predset, batch_size=1, shuffle=True)




class NaturalSceneClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(36864, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, xb):
        return self.network(xb)


model = NaturalSceneClassification()
model.load_state_dict(torch.load(f'model_intel_image.pt'))
model.to(cuda_device)
model.eval()

for features, _ in itertools.islice(pred_loader, 10):
    # print(features.shape)
    out = model(features.to(cuda_device))
    _, pred = torch.max(out.to(cpu_device), dim=1)
    # print(pred.numpy())
    # print(labels_map[pred.numpy()[0]])
    plt.imshow(features.squeeze(0).permute(1, 2, 0))
    plt.title(labels_map[pred.numpy()[0]])
    plt.show()
