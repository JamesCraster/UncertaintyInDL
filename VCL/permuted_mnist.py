"""Load in permuted mnist from a pickle to ensure consistency with MVG-VI"""
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

IMAGE_SIZE = 784

with open('../GenerateData/permuted_mnist/permuted_mnist_0.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

xtrain = []
ytrain = []
xtest = []
ytest = []
tasks = []

for data in enumerate(loaded_data):
    xtrain = torch.tensor(data[1][0])
    ytrain = torch.tensor(data[1][1])
    xtest = torch.tensor(data[1][2])
    ytest = torch.tensor(data[1][3])
    train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=256, shuffle=True)
    tasks.append((train_loader, test_loader))

def get_permuted_mnist():
    for element in tasks:
        yield element