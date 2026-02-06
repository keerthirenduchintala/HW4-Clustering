import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
    pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # checks

        # X is a numpy array and is a 2D matrix
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError("not a 2D matrix")
        
        # Y is a numpy array and is a 1D array
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        
        if y.ndim != 1:
            raise ValueError("not a 1D matrix")
        
        #obs X == obs y
        if X.shape[0] != y.shape[0]:
            raise ValueError('different number of observations')

        # Silhouette calculation 
        ## s(i) = (bi-ai)/(max(bi,ai))
        ### b(i) = mean distance from point i to points in cluster
        ### a(i) = mean distance from point i to points in nearest cluster that is not the assigned cluster
        