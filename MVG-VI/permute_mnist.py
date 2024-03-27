import gzip
import numpy as np
import pickle as pkl
import nn_utils as nnu
import theano
from VMGNet import VMGNet
floatX = theano.config.floatX
import matplotlib.pyplot as plt
import pickle 

tasks = []
with open('../GenerateData/permuted_mnist/permuted_mnist_1.pkl', 'rb') as file:
    tasks = pickle.load(file)


nn = VMGNet(tasks[0][0].shape[0], tasks[0][0].shape[1], 10, batch_size=256, dimh=(100, 100), n_iter=2,
            logtxt='vmgnet.txt', type_init='he2', n_inducing=50, ind_noise_lvl=0.01, task_type='classification')

# test our network for catastrophic forgetting

nn._create_parameters()
nn._create_model()


for i in range(0,9):
    print(f"TASK {i}")
    
    
    output_nn = nn.fit(tasks[i][0], tasks[i][1], xvalid=None, yvalid=None, xtest=tasks[i][2], ytest=tasks[i][3], sampling_rounds=2, verbose=True)
    #print(nn.layers[0].get_priors()[0].eval())
    #nn.update_priors()
    performances = []
    for j in range(0,i+1):
        preds = nn.predict(tasks[j][2], samples=5)
        performance = 100. * ((preds == tasks[j][3]).sum() / (1. * tasks[j][3].shape[0]))
        performances.append(performance)
        print(f"Test performance on task {j}: ", performance)
    print(f"Average test performance: ", sum(performances)/len(performances))


