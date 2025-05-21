import numpy as np

def softmax(x, axis=-1):
    """Compute softmax values for each set of scores in x."""
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)  # For numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
