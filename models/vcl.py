"""Implement Variational Continuous Learning"""
import torch
import torch.nn as nn
import math

class VariationalLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(VariationalLayer, self).__init__()

        # in the paper, the prior used for weights and biases is a normal with
        # zero mean and zero variance

        # to constrain variance to be positive, store v instead where v = 2 * log(variance)

        self.prior_W_m = torch.zeros(output_size, input_size, requires_grad=False) 
        self.prior_W_v = torch.zeros(output_size, input_size, requires_grad=False)

        self.prior_b_m = torch.zeros(output_size, requires_grad=False)
        self.prior_b_v = torch.zeros(output_size, requires_grad=False)
        
        # the posterior weight and bias will be optimized to match the point estimate 
        # for the first task. Therefore they are given a standard random initialisation
        self.posterior_W_m = nn.Parameter(torch.Tensor(output_size, input_size))
        self.posterior_b_m = nn.Parameter(torch.rand(output_size))
        nn.init.kaiming_uniform_(self.posterior_W_m, a=math.sqrt(5))

        # for the first task, don't give the weights any uncertainty
        self.posterior_W_v = math.log(1e-6)
        self.posterior_b_v = math.log(1e-6)
        
    def forward(self, x):
        output = x.matmul(self.posterior_W_m.t())
        output += self.posterior_b_m
        return output

class VCL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VCL, self).__init__()
        
        self.model = nn.Sequential(
            VariationalLayer(input_size, hidden_size),
            nn.ReLU(),
            # in the paper, 2 hidden layers are used
            VariationalLayer(hidden_size, hidden_size),
            nn.ReLU(),
            VariationalLayer(hidden_size, output_size),
        )

        # coreset contains examples in episodic memory
        self.coreset = []

    def forward(self, x):
        return self.model(x)

    def loss(self, output, batch_targets):
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        # add log likelihood
        elbo = - cross_entropy_loss(output, batch_targets)
        
        # the mean-field assumption is that all weights independent
        # therefore for the KL regularisation, can simply add the KL of each layer

        # pytorch minimizes, so multiply by -1
        return - elbo

    def prediction(self, input, num_samples=100):
        x = self(input)
        for i in range(0, num_samples-1):
            # for now, just use model forward
            x += self(input)
        return x/num_samples

    def update_coreset():
        # TODO use the coreset properly
        pass
