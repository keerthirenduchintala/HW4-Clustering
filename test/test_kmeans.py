# Write your k-means unit tests here
from cluster import KMeans
import pytest
import numpy as np

def test_invalid_k_value():
    with pytest.raises(ValueError):
         KMeans(k=0)

def test_predict_returns_correct_shape():
    X = np.array([[1, 2], [1, 3], [5, 6], [5, 7]])
    km = KMeans(k=2)
    km.fit(X)
    labels = km.predict(X)
    assert labels.shape == (4,)

def test_get_centroids_shape():
    X = np.array([[1, 2], [1, 3], [5, 6], [5, 7]])
    km = KMeans(k=2)
    km.fit(X)
    centroids = km.get_centroids()
    assert centroids.shape == (2, 2)
    
def test_get_error_is_number():
    X = np.array([[1, 2], [1, 3], [5, 6], [5, 7]])
    km = KMeans(k=2)
    km.fit(X)
    error = km.get_error()
    assert error >= 0