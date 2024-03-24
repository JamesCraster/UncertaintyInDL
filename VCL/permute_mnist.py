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
initial_tasks = []

for data in enumerate(loaded_data):
    xtrain = torch.tensor(data[1][0])
    ytrain = torch.tensor(data[1][1])
    xtest = torch.tensor(data[1][2])
    ytest = torch.tensor(data[1][3])

    train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=256, shuffle=True)
    initial_tasks.append((train_loader, test_loader))

def get_permuted_mnist():
    for element in initial_tasks:
        yield element

import torchvision
import torch
import torch.nn as nn
from models.vcl import VCL
from models.basic_nn import BasicNN

NUM_TASKS = 10
EPOCHS_PER_TASK = 30

def train_nn(model, tasks):
    ## a hack specific to MFVI in which you have to train the means for the weights on
    ## the first task, keeping variance zero
    ## then variance will be initialised to 10^-6 for the next task
    ## see "Radial Bayesian Neural Networks: Beyond Discrete Support In Large-Scale Bayesian Deep Learning"
    #print(f'Training the means for weights')
    #train_dataset, test_dataset = get_permuted_mnist()
    #optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    ## Performance improves when this mean initialisation is skipped. Therefore this is skipped.
    ## This is actually expected according to : Improving and Understanding Variational Continual Learning

    # for epoch in range(EPOCHS_PER_TASK):
    #     epoch_loss = 0
    #     for batch_inputs, batch_targets in train_dataset:
    #         optimizer.zero_grad()
    #         # no reparameterisation trick or kl divergence
    #         loss = model.loss_no_reparam(batch_inputs, batch_targets)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     print(f'Epoch: {epoch}, Loss: {epoch_loss/EPOCHS_PER_TASK}')

    # train on the real tasks
    generator = get_permuted_mnist()

    for task in range(0, NUM_TASKS):
        print(f'Task: {task}')

        
        # generate a new permutation of the mnist data
        train_dataset, test_dataset = next(generator)

        train_set_size = len(train_dataset.dataset)

        # store both the train and test data loaders for this task
        tasks.append((train_dataset, test_dataset))

        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        for epoch in range(EPOCHS_PER_TASK):
            epoch_loss = 0
            for batch_inputs, batch_targets in train_dataset:
                optimizer.zero_grad()
                loss = model.loss(batch_inputs, batch_targets, train_set_size)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {epoch_loss/EPOCHS_PER_TASK}')
        average = 0
        for i in range(0, task+1):
            test_performance = test_nn(model, tasks[i][0])
            print(f'Testing performance on task {i} : {test_performance} ')
            average += test_performance
        print(f'Mean test performance over tasks: {average/(task + 1)}')
        model.update_prior()

        # Improving and Understanding Variational Continual Learning 
        # recommends to reset the posterior
        #model.reset_posterior()

def test_nn(model, test_dataset):
    with torch.no_grad():
        accuracies = []
        for batch_inputs, batch_targets in test_dataset:
            output = model.predict(batch_inputs)
            _, predicted_labels = torch.max(output, 1)
            accuracy = ((predicted_labels == batch_targets).sum()/predicted_labels.shape[0]).item()
            accuracies.append(accuracy)
        return sum(accuracies)/len(accuracies)

tasks = []
model = VCL(IMAGE_SIZE, 100, 10)
train_nn(model, tasks)
