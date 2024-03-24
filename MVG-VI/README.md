Implementation of MVG-VI, updated by James Craster, taken originally from below source

Theano is no longer being maintained, there exists a fork called Pytensor. Switching to Pytensor is left as future work. To run this, you will need Python no older than 3.9 and numpy no older than 1.21.5, because Theano uses some deprecated functionality of both. Running Theano in 2024 is not trivial. I used Theano 1.0.5 and scipy 1.9.1 if that helps.

I updated the code to use PermutedMNIST - using the same code from the VCL repo to generate it. 

--- Original README ----

Example implementation of the Bayesian neural network in:

***"Structured and Efficient Variational Deep Learning with Matrix Gaussian Posteriors"***, Christos Louizos & Max Welling, ICML 2016, ([https://arxiv.org/abs/1603.04733]())

This code is provided as is and will not be maintained / updated.
