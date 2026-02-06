import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # check K type
        if not isinstance(k, int):
            raise TypeError("k must be integer")
        # check k and max_iter values
        if k < 1:
            raise ValueError("k must be greater than 1")
        if max_iter < 1:
            raise ValueError("max_iter must be greater than 1")
        
        # store parameters
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        # attributes to be kept private
        self._centroids = None
        self._error = None
        self._fitted = False  # Flag to track if model has been fit


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # checks
        # check to see that mat is a an array
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be a numpy array")
        # check to see that it is a 2D matrix
        if mat.ndim != 2:
            raise ValueError("not a 2D matrix")
        # check to see that rows >= k
        if mat.shape[0] <= self.k:
            raise ValueError("observations less than number of clusters!")

        # Initialize centroids: randomly choose centroids
        init_data_points = np.random.choice(mat.shape[0], size = self.k, replace = False)

        self._centroids = mat[init_data_points]

        # loop that continues until max_iter or below error and does:
        for i in range(self.max_iter):
            # 1. Calculate distance from each data point to each centroid
            distances = cdist(mat, self._centroids)
            # 2. Assign closest centroid to each data point (these are labels)
            labels = np.argmin(distances, axis= 1)
            # 3. Calculate new centroids (mean of points in each cluster)
            centroid_list = []
            for cluster_idx in range(self.k):    # loop through 0, 1, 2
                points_in_cluster = mat[labels == cluster_idx]
                centroid = points_in_cluster.mean(axis=0)
                centroid_list.append(centroid)

            new_centroids = np.array(centroid_list)
            # 4. Calculate error (how much did centroids change? error = sum of squared differences)
            error = np.sum((self._centroids - new_centroids) ** 2)
            # 5. If error < tolerance, stop early 
            if error < self.tol:
                break
            # 6. Update centroids for next iteration
            self._centroids = new_centroids

        self._fitted = True
        self._error = error

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # checks 
        # model been fit?
        if not self._fitted:
            raise ValueError ('Model has not been fit yet!')

        # check that matrix is valid 2D array and is an array
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be a numpy array")
        
        if mat.ndim != 2:
            raise ValueError("not a 2D matrix")
        
        # check that matrix has same number of features (columns) as centroids
        if mat.shape[1] != self._centroids.shape[1]:
            raise ValueError("different number of features!")
        
        # predict part
        distances = cdist(mat, self._centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        # model been fit?
        if not self._fitted:
            raise ValueError ('Model has not been fit yet!')
        
        return self._error
    
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # model been fit?
        if not self._fitted:
            raise ValueError ('Model has not been fit yet!')
        return self._centroids