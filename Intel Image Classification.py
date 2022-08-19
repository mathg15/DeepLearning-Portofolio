# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

assert torch.cuda.is_available()
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

labels_map = {0: 'buildings',
              1: 'forest',
              2: 'glacier',
              3: 'mountain',
              4: 'sea',
              5: 'street'}

training_path = "seg_train/seg_train"
testing_path = "seg_test/seg_test"

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 100))])
trainset = datasets.ImageFolder(training_path, transform=train_transform)
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 100))])
testset = datasets.ImageFolder(testing_path, transform=test_transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

figure = plt.figure(figsize=(8, 8))
cols, row = 4, 4
for i in range(1, cols * row + 1):
    sample_idx = torch.randint(len(trainset), size=(1,)).item()
    img, label = trainset[sample_idx]
    figure.add_subplot(row, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
plt.show()


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


model = NaturalSceneClassification().to(cuda_device)
model.eval()

# Hyperparameters
N_epochs = 30
learning_rate = 1e-3

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
i = 0
for epoch in range(N_epochs):  # Loop over epochs
    running_loss = 0.0

    for features, labels in train_loader:

        # Forward Propagation
        labels_pred = model(features.to(cuda_device))

        # Loss computation
        loss = loss_function(labels_pred, labels.to(cuda_device))

        # Save loss for future analysis
        losses.append(loss.item())

        # Erase previous gradients
        optimizer.zero_grad()

        # Compute gradients (backpropagation)
        loss.backward()

        # Weight update
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:
            print('[Epoque : %d, iteration: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        i += 1

print('Training done')


def display(losses, label='Training loss function'):
    # Display loss evolution
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.plot(losses, 'r-', lw=2, label=label)
    axes.set_xlabel('N iterations', fontsize=18)
    axes.set_ylabel('Loss', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.show()


# # Display loss evolution
display(losses)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images.to(cuda_device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(cuda_device)).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images:{(correct / total) * 100:.2f}')

save_path = f'model_intel_image.pt'
torch.save(model.state_dict(), save_path)
