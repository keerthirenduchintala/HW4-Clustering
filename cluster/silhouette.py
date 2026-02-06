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
        ### a(i) = mean distance from point i to all points in cluster
        ### b(i) = mean distance from point i to all points in nearest cluster that is not the assigned cluster
        # All pairwise distances
        distances = cdist(X, X)

        silhouette_scores = []

        for i in range(X.shape[0]):
            # What cluster does point i belong to?
            cluster_i = y[i]
    
            # Calculate a(i): mean distance to other points in same cluster
            distances_to_same_cluster = []
            for j in range(X.shape[0]):
                # Skip if j is the same point as i
                if j == i:
                    continue
                # Only include if j is in the same cluster as i
                if y[j] == cluster_i:
                    distances_to_same_cluster.append(distances[i, j])
            a_i = np.mean(distances_to_same_cluster)

            # Calculate b(i): mean distance to points in NEAREST OTHER cluster
            mean_distances_to_other_clusters = []

            for cluster in np.unique(y):
            # Skip if it's the same cluster as point i
                if cluster == cluster_i:
                    continue
                
                distances_to_this_cluster = []
                for j in range(X.shape[0]):
                    if y[j] == cluster:
                        distances_to_this_cluster.append(distances[i, j])   
            # Compute mean distance 
                mean_dist = np.mean(distances_to_this_cluster)
                mean_distances_to_other_clusters.append(mean_dist)
            b_i = np.min(mean_distances_to_other_clusters)
            # Calculate s(i) = (bi-ai)/(max(bi,ai))
            s_i = (b_i-a_i)/(max(b_i, a_i))
            silhouette_scores.append(s_i)
        score = np.array(silhouette_scores)
        return score


                