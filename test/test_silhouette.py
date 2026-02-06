# write your silhouette score unit tests here
from cluster import Silhouette
import pytest
import numpy as np
from sklearn.metrics import silhouette_score


# Basic Tests
def test_X_not_numpy_array(self):
    sil = Silhouette()
    X = [[1, 2], [3, 4]]
    y = np.array([0, 1])
    with pytest.raises(TypeError):
        sil.score(X, y)
    
def test_X_not_2D(self):
    sil = Silhouette()
    X = np.array([1, 2, 3, 4])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        sil.score(X, y)

def test_y_not_numpy_array(self):
    sil = Silhouette()
    X = np.array([[1, 2], [3, 4]])
    y = [0, 1]
    with pytest.raises(TypeError):
        sil.score(X, y)
    
def test_y_not_1D(self):
    sil = Silhouette()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        sil.score(X, y)
    
def test_mismatched_lengths(self):
    sil = Silhouette()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        sil.score(X, y)

# Tests for comparing to sklearn
my_scores = Silhouette().score(X, labels)
my_mean = np.mean(my_scores)

# sklearn's implementation
sklearn_mean = silhouette_score(X, labels)

# Compare - they should be very close
assert np.isclose(my_mean, sklearn_mean)

