import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

class Kmeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.iter = None
        self.tol = tol
        self.cluster_center_history = []
        self.cluster_centers = None
        self.train_labels = None
        self.test_labels = None

    def initialize_clusters(self, X, seed=0):
        rng = np.random.default_rng(seed)
        return rng.choice(X, self.n_clusters, replace=False)

    def get_assignments(self, X, cluster_centers):
        assignments = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            assignments[i] = np.argmin(np.linalg.norm(x - cluster_centers, axis=1)) # Output form: [0,0,2,1,i ...] where i can be one of (0,1,...K) 
        
        return assignments

    def fit(self, X):
        # Initialize cluster centers
        cluster_centers = self.initialize_clusters(X, self.n_clusters)
        self.cluster_center_history.append(cluster_centers.copy())

        # Run E step and M step for <max_iter> number of iterations (The jargon 'E' tep and 'M' step is taken from Bishop (2006))
        for i in range(self.max_iter):
            print(i)
            # E step
            assignments = self.get_assignments(X, cluster_centers)

            # M step
            for k in range(self.n_clusters):
                if np.any(assignments == k):
                    cluster_centers[k] = np.mean(X[assignments == k], axis=0)
            
            print(f"Iter {i}: Cluster centers: {cluster_centers}")

            if np.all(cluster_centers == self.cluster_center_history[-1]):
                break
            
            # Record cluster centers
            self.cluster_center_history.append(cluster_centers.copy())

        
        self.iter = i
        self.cluster_centers = cluster_centers.copy()
        self.train_labels = assignments.copy()
    
    def predict(self, X):
        self.test_labels = self.get_assignments(X, self.cluster_centers)

