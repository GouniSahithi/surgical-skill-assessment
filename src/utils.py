import numpy as np

def normalize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)
