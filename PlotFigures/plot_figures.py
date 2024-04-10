"""A small script to generate the PDF plots for the paper"""
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(10, 5))



## SPLIT MNIST

FULL_MVG_VI = [100, 48.6, 33.0, 25.0, 20.0]
DIAG_MVG_VI = [100, 25.5, 25.0, 13.0, 13.0]
VCL = [100, 56.4 , 41.5 , 26.3 , 13.7]
#MVG_VI_no_prior = [50, 40, 30, 50, 60]
#VCL_no_prior = [20, 40, 30, 50, 60]
BASIC_NN = [99.9, 49.2, 33.3, 25.0, 20.0]

x_split = [1, 2, 3, 4, 5]

axs[0].plot(x_split, FULL_MVG_VI, marker='o', label='Full MVG-VI', linestyle='-', linewidth=3)
axs[0].plot(x_split, DIAG_MVG_VI, marker='o', label='Diagonal MVG-VI')
axs[0].plot(x_split, VCL, marker='o', label='VCL')
axs[0].plot(x_split, BASIC_NN, marker='o', linestyle='-.', label='Standard NN', alpha=0.7)
#axs[0].plot(x_split, MVG_VI_no_prior, marker='o')
#axs[0].plot(x_split, VCL_no_prior, marker='o')


## Permuted MNIST

FULL_MVG_VI = [96.8, 96.3, 95.6, 95.1, 94.6, 93.8, 93.2, 93.0, 92.6, 92.0]
DIAG_MVG_VI = [96.5, 92.1, 90.7, 86.6, 84.9, 81.4, 81.1, 81.0, 78.4, 76.0]
VCL = [99.7, 94.8, 92.4 , 89.0 , 88.2, 83.8, 84.1, 82.2, 81.8, 78.0]
#MVG_VI_no_prior = [50, 40, 30, 50, 60, 90, 80, 70, 60, 50]
#VCL_no_prior = [20, 40, 30, 50, 60, 90, 80, 70, 60, 50]
BASIC_NN = [99.0, 81.3, 70.4, 63.3, 57.5, 54.6, 51.2, 48.3, 40.2, 37.5 ]

x_permute = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

axs[1].plot(x_permute, FULL_MVG_VI, marker='o', label='Full MVG-VI')
axs[1].plot(x_permute, VCL, marker='o', label='VCL')
axs[1].plot(x_permute, DIAG_MVG_VI, marker='o', label='Diagonal MVG-VI')
axs[1].plot(x_permute, BASIC_NN, marker='o', linestyle='-.', label='Basic NN')
#axs[1].plot(x_permute, VCL_no_prior, marker='o')

axs[0].set_xticks(x_split)
axs[1].set_xticks(x_permute)
axs[0].set_xlabel('Task', fontsize=12)
axs[0].legend(loc='lower left')
axs[1].legend(loc='lower left')
axs[1].set_xlabel('Task', fontsize=12)
axs[0].set_ylabel('Mean Test Accuracy (%)', fontsize=12)
axs[1].set_ylabel('Mean Test Accuracy (%)', fontsize=12)
axs[0].set_title('Single-Headed Split MNIST', fontsize=14)
axs[1].set_title('Permuted MNIST', fontsize=14)
axs[0].set_ylim(0, 100)
axs[1].set_ylim(0, 100)
plt.tight_layout()
plt.show()