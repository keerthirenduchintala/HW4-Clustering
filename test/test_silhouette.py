# write your silhouette score unit tests here
from cluster import Silhouette
import pytest
import numpy as np
from sklearn.metrics import silhouette_score
from cluster.utils import make_clusters


# Basic Tests
def test_X_not_numpy_array():
    sil = Silhouette()
    X = [[1, 2], [3, 4]]
    y = np.array([0, 1])
    with pytest.raises(TypeError):
        sil.score(X, y)
    
def test_X_not_2D():
    sil = Silhouette()
    X = np.array([1, 2, 3, 4])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        sil.score(X, y)

def test_y_not_numpy_array():
    sil = Silhouette()
    X = np.array([[1, 2], [3, 4]])
    y = [0, 1]
    with pytest.raises(TypeError):
        sil.score(X, y)
    
def test_y_not_1D():
    sil = Silhouette()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        sil.score(X, y)
    
def test_mismatched_lengths():
    sil = Silhouette()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        sil.score(X, y)

# Tests for comparing to sklearn
def test_with_make_clusters():
    
    X, y = make_clusters(n=100, k=3, scale=0.5)
    
    my_scores = Silhouette().score(X, y)
    my_mean = np.mean(my_scores)
    
    sklearn_mean = silhouette_score(X, y)
    
    assert np.isclose(my_mean, sklearn_mean)

