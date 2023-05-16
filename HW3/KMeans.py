import numpy as np

class KMeans():
    # This function initializes the KMeans class
    def __init__(self, k = 3, num_iter = 1000, order = 2):
        # Set a seed for easy debugging and evaluation
        np.random.seed(42)
        
        # This variable defines how many clusters to create
        # default is 3
        self.k = k

        # This variable defines how many iterations to recompute centroids
        # default is 1000
        self.num_iter = num_iter

        # This variable stores the coordinates of centroids
        self.centers = None

        # This variable defines whether it's K-Means or K-Medians
        # an order of 2 uses Euclidean distance for means
        # an order of 1 uses Manhattan distance for medians
        # default is 2
        if order == 1 or order == 2:
            self.order = order
        else:
            raise Exception("Unknown Order")     

    # This function fits the model with input data (training)
    def fit(self, X):
        # m, n represent the number of rows (observations) 
        # and columns (positions in each coordinate)
        m, n = X.shape

        # self.centers are a 2d-array of 
        # (number of clusters, number of dimensions of our input data)
        self.centers = np.zeros((self.k, n))

        # self.cluster_idx represents the cluster index for each observation
        # which is a 1d-array of (number of observations)
        self.cluster_idx = np.zeros(m)

        ##### TODO 1 ######
        #
        # Task: initialize self.centers
        #
        # Instruction: 
        # For each dimension (feature) in X, use the 10th percentile and 
        # the 90th percentile to form a uniform distribution. Then, we will initialize 
        # the values of each center by randomly selecting values from the distributions.
        #
        # Note:
        # This method is by no means the best initialization method. However, we would
        # like you to follow our guidelines in this HW. We will ask you to discuss some better
        # initializaiton methods in the notebook.
        #
        # Hint:
        # 1. np.random.uniform(), np.percentile() might be useful
        # 2. make sure to look over its parameters if you're not sure
        ####################

        for i in range(n):
            feature = X[:,i]
            ten_percentile = np.percentile(feature,10)
            ninety_percentile = np.percentile(feature, 90)
            self.centers[:,i] = np.random.uniform(ten_percentile,ninety_percentile, size=self.k)

        for i in range(self.num_iter):
            cluster_idx = np.zeros(m)
            distances = np.zeros((m,self.k))
            for ob_ind in range(m):
                for center_ind in range(self.k):
                    distances[ob_ind][center_ind] = np.linalg.norm(X[ob_ind] - self.centers[center_ind], ord=self.order)
                cluster_idx[ob_ind] = np.argmin(distances[ob_ind])
                
            new_centers = np.zeros((self.k, n))
            for idx in range(self.k):
                cluster_coordinates = X[cluster_idx == idx]  # first cluster_idx== idx return a bool array, then X[bool_array] keeps only the row with 'True', which in this case, the row in X belongs to the current cluster ind
                if self.order == 2:
                    cluster_center = np.mean(cluster_coordinates,axis=0)
                elif self.order == 1:
                    cluster_center = np.median(cluster_coordinates,axis=0)
                new_centers[idx, :] = cluster_center

            if np.all(cluster_idx == self.cluster_idx):
                print(f"Early Stopped at Iteration {i}")
                return self
            self.centers = new_centers
            self.cluster_idx = cluster_idx
        return self


    def predict(self, X):
        m = len(X)
        distances = np.zeros((m,self.k))
        for ob_ind in range(m):
            for center_ind in range(self.k):
                distances[ob_ind][center_ind] = np.linalg.norm(X[ob_ind] - self.centers[center_ind], ord=self.order)
            self.cluster_idx[ob_ind] = np.argmin(distances[ob_ind])
        return self.cluster_idx
