import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    return np.sqrt(np.mean((pred - tar) ** 2))