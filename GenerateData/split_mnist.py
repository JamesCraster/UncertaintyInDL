"""Generate the split mnist dataset from MNIST"""
import torchvision
import torchvision.transforms as T
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle 
import json

IMAGE_SIZE = 784
transform = T.Compose([T.PILToTensor(), T.Lambda(lambda t: torch.flatten(t)), T.Lambda(lambda t: t.float())])
mnist_train = torchvision.datasets.MNIST('.', download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST('.', train=False, download=True, transform=transform)

task_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

tasks = []

for task in task_classes:
    print(task)
    train_data = []
    train_targets = []
    for data, target in mnist_train:
        if (target == task[0]) or (target == task[1]):
            train_data.append((data / 255.0).numpy())
            train_targets.append(target)
    
    test_data = []
    test_targets = []
    for data, target in mnist_test:
        if (target == task[0]) or (target == task[1]):
            test_data.append((data / 255.0).numpy())
            test_targets.append(target)

    tasks.append((np.array(train_data), np.array(train_targets), 
    np.array(test_data), np.array(test_targets)))


with open(f'./split_mnist.pkl', 'wb') as file:
    pickle.dump(tasks, file)

