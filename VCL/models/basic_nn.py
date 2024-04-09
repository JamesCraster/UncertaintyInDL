"""Implement a basic NN and check that there is catastrophic forgetting"""
import torch
import torch.nn as nn

class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def loss(self,batch_inputs, batch_targets, train_set_size):
        loss_fn = torch.nn.CrossEntropyLoss()
        cross_entropy_loss = loss_fn(self(batch_inputs), batch_targets)
        return cross_entropy_loss

    def predict(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)
    
    def update_prior(self):
        pass

    def reset_posterior(self):
        pass
