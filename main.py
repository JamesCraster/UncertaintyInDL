import torchvision
import torch
import torch.nn as nn
from permuted_mnist import get_permuted_mnist, IMAGE_SIZE
from models.vcl import VCL
from models.basic_nn import BasicNN
import matplotlib.pyplot as plt
from torchviz import make_dot

NUM_TASKS = 1
EPOCHS_PER_TASK = 1000

tasks = []
model = BasicNN(IMAGE_SIZE, 20, 10)
def train_basic_nn(model, tasks):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # run training loop
    losses = []
    for task in range(0, NUM_TASKS):
        # generate a new permutation of the mnist data
        train_dataset, test_dataset = get_permuted_mnist()
        # store both the train and test data loaders for this task
        tasks.append((train_dataset, test_dataset))
        for epoch in range(EPOCHS_PER_TASK):
            for batch_inputs, batch_targets in train_dataset:
                output = model(batch_inputs)
                loss = model.loss(output, batch_targets)
                #loss = loss_fn(output, batch_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 20 == 0:
                    print(f"Loss is: {loss.item()}")
                    losses.append(loss.item())

def test_basic_nn(model, test_dataset):
    with torch.no_grad():
        accuracies = []
        for batch_inputs, batch_targets in test_dataset:
            output = model(batch_inputs)
            _, predicted_labels = torch.max(output, 1)
            accuracy = ((predicted_labels == batch_targets).sum()/predicted_labels.shape[0]).item()
            accuracies.append(accuracy)
        return sum(accuracies)/len(accuracies)



def train_vcl(model, tasks):
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    # run training loop
    losses = []
    
    model = BasicNN(IMAGE_SIZE, 20, 10)

    loss_fn = nn.CrossEntropyLoss()

    for task in range(0, NUM_TASKS):
        # generate a new permutation of the mnist data
        train_dataset, test_dataset = get_permuted_mnist()
        # store both the train and test data loaders for this task
        tasks.append((train_dataset, test_dataset))
        for x in train_dataset:
            batch_inputs, batch_targets = x
            break
        for epoch in range(EPOCHS_PER_TASK):
            
            optimizer.zero_grad()
            output = model(batch_inputs)
            # compute the ELBO and take a step to maximise it
            #loss = model.loss(output, batch_targets)
            loss = loss_fn(output, batch_targets)
            # print(loss)
            # make_dot(batch_inputs).view()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"Loss is: {loss.item()}")
                losses.append(loss.item())
                output = model(batch_inputs)
                _, predicted_labels = torch.max(output, 1)
                accuracy = ((predicted_labels == batch_targets).sum()/predicted_labels.shape[0]).item()
                print(accuracy)
        # replace the prior with the current posterior
    plt.plot(losses)
    plt.show()

def test_vcl(model, test_dataset):
    with torch.no_grad():
        accuracies = []
        for batch_inputs, batch_targets in test_dataset:
            output = model.prediction(batch_inputs)
            _, predicted_labels = torch.max(output, 1)
            accuracy = ((predicted_labels == batch_targets).sum()/predicted_labels.shape[0]).item()
            accuracies.append(accuracy)
        return sum(accuracies)/len(accuracies)

#model = VCL(IMAGE_SIZE, 20, 10)
#tasks = []
#train_vcl(model, tasks)



            






