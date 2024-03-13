"""Implement Variational Continuous Learning"""
import torch
import torch.nn as nn
import math

# class VariationalLayer(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(VariationalLayer, self).__init__()
        
#         # in the paper, the prior used for weights and biases is a normal with 
#         # zero mean and zero variance

#         # to constrain variance to be positive, store p instead where p = 2 * log(variance)

#         self.prior_W_mean = torch.zeros(output_size, input_size, requires_grad=False) 
#         self.prior_W_p = torch.zeros(output_size, input_size, requires_grad=False)

#         self.prior_b_mean = torch.zeros(output_size, requires_grad=False)
#         self.prior_b_p = torch.zeros(output_size, requires_grad=False)

#         self.posterior_W_mean = nn.Parameter(torch.rand(output_size, input_size))
#         torch.nn.init.uniform_(self.posterior_W_mean)
        
#         self.posterior_W_p = nn.Parameter(torch.zeros(output_size, input_size))

#         self.posterior_b_mean = nn.Parameter(torch.rand(output_size))
#         torch.nn.init.uniform_(self.posterior_b_mean)

#         self.posterior_b_p = nn.Parameter(torch.zeros(output_size))

#     def kl_divergence():
#         # closed form of the KL divergence is known for multivariate gaussians
#         return 
        
    
#     def forward(self, x):
#         """
#         Generate a training forward pass by sampling from the posterior distribution
#         NOT to be used for test predictions
#         """
#         # sample from the weights and biases distribution
#         # use the bayes-by-backprop aka reparameterisation trick
#         weight_shape = self.posterior_W_mean.shape
#         weight_epsilons = torch.normal(torch.zeros(weight_shape), torch.zeros(weight_shape))
#         weights = self.posterior_W_mean #+ torch.exp(self.posterior_W_p) * weight_epsilons
#         bias_shape = self.posterior_b_mean.shape
#         bias_epsilons = torch.normal(torch.zeros(bias_shape), torch.zeros(bias_shape))
#         biases = self.posterior_b_mean #+ torch.exp(self.posterior_b_p) * bias_epsilons
#         return (weights @ x.T).T + biases

class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(VariationalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.posterior_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.posterior_bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = x.matmul(self.posterior_weight.t())
        output += self.bias
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
        elbo = cross_entropy_loss(output, batch_targets)
        # the mean-field assumption is that all weights independent
        # therefore for the KL regularisation, can simply add the KL of each layer

        # pytorch minimizes, so multiply by -1
        return elbo

    def prediction(self, input, num_samples=100):
        x = self(input)
        for i in range(0, num_samples-1):
            # for now, just use model forward
            x += self(input)
        return x/num_samples

    def update_coreset():
        # TODO use the coreset properly
        pass
