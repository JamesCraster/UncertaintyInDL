"""Generate the standard permuted mnist dataset from MNIST"""
import torchvision
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle 
import json

IMAGE_SIZE = 784
transform = T.Compose([T.PILToTensor(), T.Lambda(lambda t: torch.flatten(t)), T.Lambda(lambda t: t.float())])
mnist_train = torchvision.datasets.MNIST('.', download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST('.', train=False, download=True, transform=transform)

def permute(dataloader, permutation):
    xtrain_numpy = []
    ytrain_numpy = []
    xtrain_torch = []
    ytrain_torch = []
    for data in dataloader:
        image = torch.index_select(data[0], dim=0, index=permutation)
         # now scale the data (max brightness is 255)
        image = image / 255.0
        xtrain_numpy.append(image.numpy())
        ytrain_numpy.append(data[1])
    return np.array(xtrain_numpy), np.array(ytrain_numpy)

def get_permuted_mnist():
    permutation = torch.randperm(IMAGE_SIZE)
    xtrain_numpy, ytrain_numpy = permute(mnist_train, permutation)
    xtest_numpy, ytest_numpy = permute(mnist_test, permutation)
    return (permutation, (xtrain_numpy, ytrain_numpy, xtest_numpy, ytest_numpy))

for file_num in range(0, 3):
    tasks = []
    permutations = []
    print(f"Generating pickle file: {file_num}")
    for i in range(0,10):
        print(f"Generating task: {i}")
        permutation, permuted_data = get_permuted_mnist()
        permutations.append(permutation.tolist())
        tasks.append(permuted_data)

    with open(f'permuted_mnist/permuted_mnist_{file_num}.pkl', 'wb') as file:
        # output the permutations to JSON
        # pickle the numpy version
        # pickle to torch version
        pickle.dump(tasks, file)
    
    with open(f'permuted_mnist/permuted_mnist_{file_num}.json', 'w') as file:
        # output the permutations to JSON
        # pickle the numpy version
        # pickle to torch version
        json.dump(permutations, file)


