import numpy as np
from scipy.stats import norm

# N(0, 1) generator
N_01  = norm(0, 1)
# Normal PDF at 0
NPDF_0 = N_01.pdf(0) + 0.05

def h(scores):
    return [(1 - n) * s for s, n in zip(scores, normals(len(scores)))]

def normals(n):
    if n <= 0: return np.array([])
    if n == 1: return np.array([0])
    if n == 2: return np.array([0, 0])
    # Center
    bins = np.arange(0, n) - n//2
    # Normalize
    bins = bins / n
    # Extend until 3 variances
    bins = bins * 4
    # N(0, 1) PDF
    n = N_01.pdf(bins)
    # Normalize
    return n / NPDF_0
