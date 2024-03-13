"""Generate the standard permuted mnist dataset from MNIST"""
import torchvision
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader

IMAGE_SIZE = 784
transform = T.Compose([T.PILToTensor(), T.Lambda(lambda t: torch.flatten(t)), T.Lambda(lambda t: t.float())])
mnist_train = torchvision.datasets.MNIST('.', download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST('.', train=False, download=True, transform=transform)

def permute(dataloader, permutation):
    output = []
    for data in dataloader:
        output.append((torch.index_select(data[0], dim=0, index=permutation), data[1]))
    return output

def get_permuted_mnist():
    permutation = torch.randperm(IMAGE_SIZE)
    train_data = DataLoader(permute(mnist_train, permutation), batch_size=256, shuffle=True)
    test_data = DataLoader(permute(mnist_test, permutation), batch_size=256, shuffle=True)
    return train_data, test_data