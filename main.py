import torchvision
import torch
import torch.nn as nn
from permuted_mnist import get_permuted_mnist, IMAGE_SIZE
from models.vcl import VCL
from models.basic_nn import BasicNN
import matplotlib.pyplot as plt
from torchviz import make_dot

NUM_TASKS = 3
EPOCHS_PER_TASK = 50

def train_nn(model, tasks):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)

    # run training loop
    losses = []
    for task in range(0, NUM_TASKS):
        print(f'Task: {task}')
        # generate a new permutation of the mnist data
        train_dataset, test_dataset = get_permuted_mnist()
        # store both the train and test data loaders for this task
        tasks.append((train_dataset, test_dataset))
        for epoch in range(EPOCHS_PER_TASK):
            epoch_loss = 0
            for batch_inputs, batch_targets in train_dataset:
                include_kl = task != 0
                optimizer.zero_grad()
                loss = model.loss(batch_inputs, batch_targets, len(train_dataset), include_kl)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {epoch_loss/EPOCHS_PER_TASK}')
            if loss.item() < 0.1:
                model.update_prior()
                break
        for i in range(0, task+1):
            print(f'Testing performance on task {i} : {test_nn(model, tasks[i][0])} ')
        model.update_prior()

    plt.plot(losses)
    plt.show()

def test_nn(model, test_dataset):
    with torch.no_grad():
        accuracies = []
        for batch_inputs, batch_targets in test_dataset:
            output = model(batch_inputs)
            _, predicted_labels = torch.max(output, 1)
            accuracy = ((predicted_labels == batch_targets).sum()/predicted_labels.shape[0]).item()
            accuracies.append(accuracy)
        return sum(accuracies)/len(accuracies)

tasks = []
model = VCL(IMAGE_SIZE, 100, 10)
train_nn(model, tasks)
