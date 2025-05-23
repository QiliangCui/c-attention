import numpy as np
from cattention import softmax

def test_softmax_1d():
    x = np.array([1.0, 2.0, 3.0])
    y = softmax(x)
    assert np.allclose(np.sum(y), 1.0)
    assert y.shape == x.shape

def test_softmax_2d():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y = softmax(x, axis=1)
    assert np.allclose(np.sum(y, axis=1), np.ones(x.shape[0]))
    assert y.shape == x.shape
