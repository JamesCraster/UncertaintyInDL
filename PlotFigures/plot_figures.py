"""A small script to generate the PDF plots for the paper"""
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(10, 5))



## SPLIT MNIST

MVG_VI = [90, 80, 70, 60, 50]
VCL = [100, 56.4 , 41.5 , 26.3 , 13.7]
MVG_VI_no_prior = [50, 40, 30, 50, 60]
VCL_no_prior = [20, 40, 30, 50, 60]

x_split = [1, 2, 3, 4, 5]


axs[0].plot(x_split, MVG_VI, marker='o')
axs[0].plot(x_split, VCL, marker='o')
axs[0].plot(x_split, MVG_VI_no_prior, marker='o')
axs[0].plot(x_split, VCL_no_prior, marker='o')


## Permuted MNIST

MVG_VI = [90, 80, 70, 60, 50, 90, 80, 70, 60, 50]
VCL = [99.7, 94.8, 92.4 , 89.0 , 88.2, 83.8, 84.1, 82.2, 81.8, 78.0]
MVG_VI_no_prior = [50, 40, 30, 50, 60, 90, 80, 70, 60, 50]
VCL_no_prior = [20, 40, 30, 50, 60, 90, 80, 70, 60, 50]

x_permute = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

axs[1].plot(x_permute, MVG_VI, marker='o')
axs[1].plot(x_permute, VCL, marker='o')
axs[1].plot(x_permute, MVG_VI_no_prior, marker='o')
axs[1].plot(x_permute, VCL_no_prior, marker='o')

axs[0].set_xticks(x_split)
axs[1].set_xticks(x_permute)
axs[0].set_xlabel('Task', fontsize=12)
axs[1].set_xlabel('Task', fontsize=12)
axs[0].set_ylabel('Mean Test Accuracy (%)', fontsize=12)
axs[1].set_ylabel('Mean Test Accuracy (%)', fontsize=12)
axs[0].set_title('Single-Headed Split MNIST', fontsize=14)
axs[1].set_title('Permuted MNIST', fontsize=14)

plt.tight_layout()
plt.show()