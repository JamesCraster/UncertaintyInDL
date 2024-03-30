"""Implement Matrix Variate Gaussian VI with a full U and V"""
import torch
import torch.nn as nn
import math

def cholesky_compose(U_half, mask):
    return (U_half * mask) @ (U_half * mask).T

def cholesky_log_det(U_half):
    return (torch.log(torch.pow(torch.diag(U_half),2))).sum()

def MVG_KL(M_post, M_prior, U_post_half, U_prior_half, U_mask, V_post_half, V_prior_half, V_mask, input_size, output_size):
    U_post = cholesky_compose(U_post_half, U_mask)
    V_post = cholesky_compose(V_post_half, V_mask)
    
    U_prior_inv = torch.cholesky_inverse(U_prior_half * U_mask)
    V_prior_inv = torch.cholesky_inverse(V_prior_half * V_mask)

    first_term = torch.trace(U_prior_inv @ U_post) * torch.trace(V_prior_inv @ V_post)
    
    middle_term = ((M_prior - M_post).T @ U_prior_inv) @ ((M_prior - M_post) @ V_prior_inv)
    
    last_term = - output_size * input_size \
        + output_size * (cholesky_log_det(U_prior_half * U_mask) - cholesky_log_det(U_post_half * U_mask)) \
            + input_size * (cholesky_log_det(V_prior_half * V_mask) - cholesky_log_det(V_post_half * V_mask))

    output = first_term + torch.trace(middle_term) + last_term
    
    return 0.5 * output


class MVGLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(MVGLayer, self).__init__()

        self.output_size = output_size
        # add an extra column for the bias term
        self.input_size = input_size + 1

        # in the paper, the prior used for weights and biases is a normal with
        # zero mean and zero variance
        # to constrain variance to be positive, store v instead where v = log(variance)

        self.prior_W_m = torch.zeros(self.input_size, self.output_size, requires_grad=False) 
        self.prior_W_u = torch.full((self.input_size,self.input_size), 0.0)
        self.prior_W_u.fill_diagonal_(1.0)
        self.prior_W_v = torch.full((self.output_size,self.output_size), 0.0)
        self.prior_W_v.fill_diagonal_(1.0)

        self.reset_posterior()
               
    def forward(self, x):
        # add input column of ones to provide for a bias term
        x = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
        
        # perform the reparameterisation trick

        U_half = (self.posterior_W_u * self.posterior_W_u_mask)
        V_half = (self.posterior_W_v * self.posterior_W_v_mask).T
        weight_epsilons = torch.normal(mean=0, std=1, size=self.posterior_W_m.shape)
        
        output = x.matmul(self.posterior_W_m + U_half @ weight_epsilons @ V_half)

        return output

    def forward_no_reparam(self,x):
        # do standard forward pass without any variance

        output = x.matmul((self.posterior_W_m).t())

        return output

    def kl_divergence(self):
        # compute the kl divergence between the posterior and the prior
        # remember when the posterior and prior are equal (at start of new task)
        # this KL should equal zero
        # since all the weights are independent, this is effectively just summing
        # the KL for each weight

        # This can be done in pytorch using MultivariateNormal and torch.distributions.kl.kl_divergence, 
        # but chose to implement from scratch

        return MVG_KL(self.posterior_W_m, self.prior_W_m, 
        self.posterior_W_u,
        self.prior_W_u,
        self.posterior_W_u_mask,
        self.posterior_W_v, 
        self.prior_W_v, 
        self.posterior_W_v_mask,
        self.input_size, self.output_size)
        
    def update_prior(self):
        # update the prior to be the current posterior
        self.prior_W_m = self.posterior_W_m.detach().clone()
        self.prior_W_v = self.posterior_W_v.detach().clone()
        self.prior_W_u = self.posterior_W_u.detach().clone()
    
    def reset_posterior(self):
        print("reset posterior!")
        self.posterior_W_m = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        nn.init.normal_(self.posterior_W_m, mean=0.0, std=1e-3)

        # give the weights low variance
        # in addition to storing log variances, store each vector as a cholesky decomposition of the 
        # positive definite U and V matrices

        self.posterior_W_u = nn.Parameter(torch.normal(mean=1e-3, std=1e-2, size=(self.input_size,self.input_size)))
        self.posterior_W_u_mask = torch.tril(torch.ones_like(self.posterior_W_u))
        
        self.posterior_W_v = nn.Parameter(torch.normal(mean=1e-3, std=1e-2, size=(self.output_size,self.output_size)))
        self.posterior_W_v_mask = torch.tril(torch.ones_like(self.posterior_W_v))


class MVG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MVG, self).__init__()

        self.training_samples = 3
        self.testing_samples = 10
        
        self.variational_layers = nn.Sequential(
            MVGLayer(input_size, hidden_size),
            # in the original paper, 2 hidden layers are used
            MVGLayer(hidden_size, hidden_size),
            MVGLayer(hidden_size, hidden_size),
            MVGLayer(hidden_size, output_size)
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

    def forward_no_reparam(self, x):
        relu = nn.ReLU()
        first_layer = True
        for layer in self.variational_layers:
            if not first_layer:
                x = relu(x)
            x = layer.forward_no_reparam(x)
            first_layer = False
        return x

    def loss_no_reparam(self, batch_inputs, batch_targets):
        loss_fn = torch.nn.CrossEntropyLoss()
        cross_entropy_loss = loss_fn(self.forward_no_reparam(batch_inputs), batch_targets)
        return cross_entropy_loss
   
    def loss(self, batch_inputs, batch_targets, training_set_size):
        loss_fn = torch.nn.CrossEntropyLoss()

        cross_entropy_loss = loss_fn(self(batch_inputs), batch_targets)
        for i in range(0, self.training_samples - 1):
            cross_entropy_loss += loss_fn(self(batch_inputs), batch_targets)
        
        # average the likelihood over many training samples
        # this is Monte Carlo integration
        elbo = cross_entropy_loss / self.training_samples


        kl = 0
        for layer in self.variational_layers:
            # without the kl divergence term, the model behaviour is 
            # empirically indistinguishable from a basic neural network
            # with L2 regularisation

            # divide by training set (i.e task) size, because in the paper equation (4),
            # the likelihood term is summed from n=1 to N_t (the size of task)
            # here we have taken the mean expected likelihood for the batch and 
            # used it to estimate the mean expected likelihood for the whole task
            # therefore we have divided through the eqn from (4) by N_t
            kl += layer.kl_divergence() / training_set_size

        elbo += kl
        return elbo

    def predict(self, input):
        x = self(input)
        for i in range(0, self.testing_samples-1):
            # use forward because need to take into account variance of each node
            x += self(input)
        
        return x/self.testing_samples

    def update_prior(self):
        for layer in self.variational_layers:
            layer.update_prior()

    def reset_posterior(self):
        for layer in self.variational_layers:
            layer.reset_posterior()

    def update_coreset():
        # TODO use the coreset properly
        pass
