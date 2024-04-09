Uncertainty in DL Miniproject

Contains my own implementation of VCL, diagonal MVG-VI, and MVG-VI.

Cited sources:
I looked at the reference implementation (https://github.com/nvcuong/variational-continual-learning, Cuong V. Nguyen et al., Variational Continual Learning) for inspiration for the VCL code, however I stress that all of this code is my own.
Especially, note that the reference MVG code that exists (https://github.com/AMLab-Amsterdam/SEVDL_MGP, Christos Louizos et al., Structured and Efficient Variational Deep Learning with Matrix Gaussian Posteriors):
1. Does not include full MVG
2. Is not set up for continual learning
3. Is written in Theano, which is deprecated 
4. Uses a highly complex pseudo data method for an analogy to GPs

Therefore my implementation of MVG-VI is very novel.

See the Generate Data folder for permuted MNIST. 
For permuted MNIST, their permutations are stored in JSON, so you can reproduce the datasets exactly. There is code to generate the pkl files, but these are too large to commit (without using LFS)