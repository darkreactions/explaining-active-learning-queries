"""
Pearson VII Function Kernel
Author: Richard Phillips
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist

def PUK_kernel(X1,X2, sigma=7, omega=0.5):
    # Compute the kernel matrix between two arrays using the Pearson VII function-based universal kernel.

    # Compute squared euclidean distance between each row element pair of the two matrices
    if X1 is X2 :
        kernel = squareform(pdist(X1, 'sqeuclidean'))
    else:
        kernel = cdist(X1, X2, 'sqeuclidean')

    kernel = (1 + (kernel * 4 * np.sqrt(2**(1.0/omega)-1)) / sigma**2) ** omega
    kernel = 1/kernel

    return kernel
