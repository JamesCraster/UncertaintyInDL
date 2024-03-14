"""Implement Variational Continuous Learning"""
import torch
import torch.nn as nn
import math

class VariationalLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(VariationalLayer, self).__init__()

        # in the paper, the prior used for weights and biases is a normal with
        # zero mean and zero variance

        # to constrain variance to be positive, store v instead where v = log(variance)

        self.prior_W_m = torch.zeros(output_size, input_size, requires_grad=False) 
        self.prior_W_v = torch.zeros(output_size, input_size, requires_grad=False)

        self.prior_b_m = torch.zeros(output_size, requires_grad=False)
        self.prior_b_v = torch.zeros(output_size, requires_grad=False)
        
        # the posterior weight and bias will be optimized to match the point estimate 
        # for the first task. Therefore they are given a standard random initialisation
        self.posterior_W_m = nn.Parameter(torch.Tensor(output_size, input_size))
        self.posterior_b_m = nn.Parameter(torch.rand(output_size))
        nn.init.kaiming_uniform_(self.posterior_W_m, a=math.sqrt(5))

        # for the first task, give the weights low variance
        self.posterior_W_v = torch.full((output_size, input_size), math.log(1e-6))
        self.posterior_b_v = torch.full((output_size,), math.log(1e-6))
               
    def forward(self, x):
        # perform the reparameterisation trick
        weight_epsilons = torch.normal(mean=0, std=1, size=self.posterior_W_m.shape)
        output = x.matmul((self.posterior_W_m + torch.exp(0.5 * self.posterior_W_v) * weight_epsilons).t())
        
        bias_epsilons = torch.normal(mean=0, std=1, size=self.posterior_b_m.shape)

        output += self.posterior_b_m + (torch.exp(0.5 * self.posterior_b_v) * bias_epsilons).t()
        return output
    
    def _kl_divergence(self):
        # compute the kl divergence between the posterior and the prior
        
        alpha = (1/torch.exp(self.prior_W_v)) 
        # remember when the posterior and prior are equal (at start of new task)
        # this KL should equal zero
        # since all the weights are independent, this is effectively just summing
        # the KL for each weight
        output = (torch.exp(self.posterior_W_v)/torch.exp(self.prior_W_v)  \
        + torch.pow((self.posterior_W_m - self.prior_W_m), 2)/torch.exp(self.prior_W_v) - 1 \
            + self.prior_W_v - self.posterior_W_v ).sum()
        
        return 0.5 * output
        

    def update_prior(self):
        # update the prior to be the current posterior
        self.prior_W_m = self.posterior_W_m.detach().clone()
        self.prior_W_v = self.posterior_W_v.detach().clone()
        self.prior_b_m = self.posterior_b_m.detach().clone()
        self.prior_b_v = self.posterior_b_v.detach().clone()

class VCL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VCL, self).__init__()

        self.training_samples = 10
        
        self.variational_layers = nn.Sequential(
            VariationalLayer(input_size, hidden_size),
            # in the original paper, 2 hidden layers are used
            VariationalLayer(hidden_size, hidden_size),
            VariationalLayer(hidden_size, hidden_size),
            VariationalLayer(hidden_size, output_size)
        )

        # coreset contains examples in episodic memory
        self.coreset = []

    def forward(self, x):
        relu = nn.ReLU()
        first_layer = True
        for layer in self.variational_layers:
            if not first_layer:
                x = relu(x)
            x = layer(x)
            first_layer = False
        return x

    def loss(self, batch_inputs, batch_targets, training_set_size, include_kl = True):
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(self(batch_inputs), batch_targets)
        # add log likelihood
        if include_kl:
            print('cross entropy loss', cross_entropy_loss)

        elbo = - cross_entropy_loss
        #for i in range(0, self.training_samples-1):
        #    elbo -= cross_entropy_loss(self(batch_inputs), batch_targets)
        #elbo /= self.training_samples
        
        # the mean-field assumption is that all weights independent
        # therefore for the KL regularisation, can simply add the KL of each layer
        if include_kl:
            for layer in self.variational_layers:
                # without the kl divergence term, the model behaviour is 
                # empirically indistinguishable from a basic neural network
                print(f'layer kl, {layer._kl_divergence()}')
                # TODO - why do I have to divide KL by such a large number???
                elbo -= layer._kl_divergence() #/ 1000000
                
        # pytorch minimizes, so multiply by -1
        return -elbo

    def prediction(self, input, num_samples=100):
        x = self(input)
        for i in range(0, num_samples-1):
            # for now, just use model forward
            x += self(input)
        return x/num_samples

    def update_prior(self):
        for layer in self.variational_layers:
            print(layer._kl_divergence())
            layer.update_prior()
            print(layer._kl_divergence())

    def update_coreset():
        # TODO use the coreset properly
        pass
