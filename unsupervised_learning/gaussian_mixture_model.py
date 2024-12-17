import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from scipy.stats import multivariate_normal
from scipy.io import loadmat

# Define custom colors
custom_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Shades of green, red, and blue
custom_cmap = ListedColormap(custom_colors)

seed = 0
class GaussianMixtureModel:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.parameters_history = {"mean_vectors": [], "cov_matrices": [], "mixing_coeffs": []}
        self.final_parameters = None
        self.train_labels = None
        self.test_labels = None
        self.likelihoods = []
        self.seed = 0
    
    def initialize(self, X):
        rng = np.random.default_rng(self.seed)
        n_features = X.shape[1]
        lower_limit, higher_limit = np.amin(X, axis=0), np.amax(X, axis=0)
        mean_vectors = rng.uniform(lower_limit, higher_limit, (self.n_clusters, n_features))
        cov_matrices = [np.cov(X, rowvar=False) for _ in range(self.n_clusters)]
        mixing_coeffs = np.full(self.n_clusters, 1 / self.n_clusters)
        return mean_vectors, cov_matrices, mixing_coeffs
    
    def calculate_responsibilities(self, X, mean_vectors, cov_matrices, mixing_coeffs):
        n_samples = X.shape[0]

        # Compute likelihood matrix
        likelihood_matrix = np.zeros((self.n_clusters, n_samples))
        for k in range(self.n_clusters):
            likelihood_matrix[k, :] = multivariate_normal.pdf(X, mean=mean_vectors[k], cov=cov_matrices[k])
        
        # Weighted likelihood and evidence
        weighted_likelihood = likelihood_matrix * mixing_coeffs[:, np.newaxis] # likelihood_matrix (k, n) * mixing_coeffs (k)
        evidence_vector = weighted_likelihood.sum(axis=0, keepdims=True)
        
        # Responsibility matrix
        responsibility_matrix = weighted_likelihood / evidence_vector

        return responsibility_matrix, evidence_vector

    def calculate_log_likelihood(self, evidence_vector):
        log_evidence = np.log(evidence_vector)
        log_likelihood = np.sum(log_evidence)
        return log_likelihood

    def calculate_covariances_vectorised(self, X, responsibility_matrix, mean_vectors):
        """
        Calculate the covariance matrices for each cluster in a Gaussian Mixture Model (GMM) using a vectorized approach.

        This method computes the covariance matrices based on the weighted outer products of the differences 
        between the data points and the corresponding cluster mean vectors. The responsibilities of each 
        data point for each cluster are used to weight these outer products.

        Parameters:
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data points for which the covariance matrices are to be calculated.

        responsibility_matrix : np.ndarray, shape (n_clusters, n_samples)
            The responsibility matrix indicating the degree to which each data point belongs to each cluster.
            Each entry represents the probability of a data point belonging to a specific cluster.

        mean_vectors : np.ndarray, shape (n_clusters, n_features)
            The mean vectors for each cluster. Each row corresponds to the mean of a specific cluster.

        Returns:
        -------
        cov_matrices : np.ndarray, shape (n_clusters, n_features, n_features)
            The computed covariance matrices for each cluster. Each matrix represents the covariance of the 
            data points assigned to that cluster, reflecting the spread and orientation of the data.

        Notes:
        -----
        The method uses the following steps:
        1. Computes the difference between each data point and the corresponding cluster mean vector, resulting 
        in a 3D tensor of shape (n_samples, n_clusters, n_features).
        2. Calculates the outer products of these difference vectors to capture the relationships between features
        resulting a tensor of shape (n_samples, n_clusters, n_features, n_features)
        3. Weights the outer products by the responsibilities to account for the contribution of each data point 
        to the cluster.
        4. Sums the weighted outer products across all samples for each cluster and normalizes by the total 
        responsibilities to obtain the average covariance matrix for each cluster.
        """
        diff_tensor = X[:, np.newaxis, :] - mean_vectors[np.newaxis, :, :] # X.shape = (n, m), # mean_vectors.shape = (k,m). 
        outer_products = np.einsum('...i,...j->...ij', diff_tensor, diff_tensor) # Outerproduct of every diff vector with itself
        weighted_outer_products = responsibility_matrix.T[:, :, np.newaxis, np.newaxis] * outer_products # responsibility_matrix.shape = (k,n) and outerproducts.shape = (n,k,m,m)
        cov_matrices = weighted_outer_products.sum(axis=0) / responsibility_matrix.sum(axis=1)[:, np.newaxis, np.newaxis] # the denominator is the Nk_vector
        return cov_matrices

    def e_step(self, X, mean_vectors, cov_matrices, mixing_coeffs):
        responsibility_matrix, evidence_vector = self.calculate_responsibilities(X, mean_vectors, cov_matrices, mixing_coeffs)
        log_likelihood = self.calculate_log_likelihood(evidence_vector)
        return responsibility_matrix, log_likelihood

    def m_step(self, X, responsibility_matrix):
        Nk_vector = responsibility_matrix.sum(axis=1)  # Total responsibilities per cluster
        mean_vectors = np.dot(responsibility_matrix, X) / Nk_vector[:, np.newaxis]
        cov_matrices = self.calculate_covariances_vectorised(X, responsibility_matrix, mean_vectors)
        mixing_coeffs = Nk_vector / responsibility_matrix.shape[1]  # Normalize responsibilities
        return mean_vectors, cov_matrices, mixing_coeffs

    def fit(self, X, tol=1e-5):
        # Initialize parameters
        mean_vectors, cov_matrices, mixing_coeffs = self.initialize(X)
        self.parameters_history["mean_vectors"].append(mean_vectors)
        self.parameters_history["cov_matrices"].append(cov_matrices)
        self.parameters_history["mixing_coeffs"].append(mixing_coeffs)

        log_likelihoods = []

        for i in range(self.max_iter):
            # E-step
            responsibility_matrix, log_likelihood = self.e_step(X, mean_vectors, cov_matrices, mixing_coeffs)
            log_likelihoods.append(log_likelihood)

            # Check convergence
            if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                print(f"Converged at iteration {i}.")
                break

            # M-step
            mean_vectors, cov_matrices, mixing_coeffs = self.m_step(X, responsibility_matrix)

            # Save parameter history
            self.parameters_history["mean_vectors"].append(mean_vectors)
            self.parameters_history["cov_matrices"].append(cov_matrices)
            self.parameters_history["mixing_coeffs"].append(mixing_coeffs)

        self.final_parameters = {"mean_vectors": mean_vectors, "cov_matrices": cov_matrices, "mixing_coeffs": mixing_coeffs}
        self.likelihoods = log_likelihoods

    def predict(self, X):
        """
        Predict the cluster labels for each data point in the input data.

        This method computes the responsibility matrix using the final parameters obtained from fitting 
        the Gaussian Mixture Model (GMM) to the training data. It then assigns each data point to the 
        cluster with the highest responsibility.

        Parameters:
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data points for which to predict cluster labels.

        Returns:
        -------
        labels : np.ndarray, shape (n_samples,)
            Predicted cluster labels for each data point. Each label corresponds to the cluster index 
            that the data point is most likely to belong to.

        responsibility_matrix : np.ndarray, shape (n_clusters, n_samples)
            The responsibility matrix indicating the degree to which each data point belongs to each cluster.
            Each entry represents the probability of a data point belonging to a specific cluster.

        Notes:
        -----
        The method uses the final parameters (mean vectors, covariance matrices, and mixing coefficients) 
        obtained from the fitting process to compute the responsibilities. The cluster label for each 
        data point is determined by finding the cluster with the maximum responsibility.

        This method is useful for assigning new data points to the learned clusters after the model has 
        been trained.
        """        # Compute the responsibility matrix using final parameters
        responsibility_matrix, _ = self.calculate_responsibilities(
            X,
            self.final_parameters["mean_vectors"],
            self.final_parameters["cov_matrices"],
            self.final_parameters["mixing_coeffs"]
        )
        
        # Assign each data point to the cluster with the highest responsibility
        labels = np.argmax(responsibility_matrix, axis=0)
        return labels, responsibility_matrix


def plot_data_colored_by_labels(X, labels, centers=None, is_train=True):
    """
    Plot the data colored by cluster labels and optionally plot the centers.

    Parameters:
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data points to be plotted.

    labels : np.ndarray, shape (n_samples,)
        The cluster labels assigned to each data point.

    centers : np.ndarray, shape (n_clusters, n_features), optional
        The mean vectors (centers) of the clusters to be plotted.

    is_train : bool, optional
        If True, indicates that the data is training data. Adjusts the title accordingly.
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=custom_cmap, marker='o', edgecolor='k', s=50)
    title = f'{"Training" if is_train else "Testing"} Data Colored by Labels'
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot the centers if provided
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centers')
        plt.legend()


def plot_data_neutral(X, is_train=True):
    """
    Plot the data in a neutral color.

    Parameters:
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data points to be plotted.

    is_train : bool, optional
        If True, indicates that the data is training data. Adjusts the title accordingly.
    """
    plt.scatter(X[:, 0], X[:, 1], color='m', marker='o', edgecolor='k', s=50)
    title = f'{"Training" if is_train else "Testing"} Data'
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')


def plot_data_colored_by_responsibilities(X, responsibility_matrix, centers=None, is_train=True):
    """
    Plot the data colored according to the responsibilities of the clusters and optionally plot the centers.

    Parameters:
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data points to be plotted.

    responsibility_matrix : np.ndarray, shape (n_clusters, n_samples)
        The responsibility matrix indicating the degree to which each data point belongs to each cluster.

    centers : np.ndarray, shape (n_clusters, n_features), optional
        The mean vectors (centers) of the clusters to be plotted.

    is_train : bool, optional
        If True, indicates that the data is training data. Adjusts the title accordingly.
    """
    # Normalize the responsibility matrix for coloring
    responsibility_colors = responsibility_matrix / responsibility_matrix.sum(axis=0)

    # Create a color for each point based on the responsibilities
    colors = responsibility_colors.T  # Transpose to get shape (n_samples, n_clusters)

    # Plot each point with the corresponding color based on responsibilities
    for i in range(colors.shape[1]):
        plt.scatter(X[:, 0], X[:, 1], 
                    c=colors[:, i],  # Use 'c' to specify the color mapping
                    cmap=custom_cmap,  # Specify the custom colormap
                    marker='o', 
                    edgecolor='k', 
                    s=50, 
                    alpha=0.5)  # Adjust alpha for transparency

    title = f'{"Training" if is_train else "Testing"} Data Colored by Responsibilities'
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot the centers if provided
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centers')
        plt.legend()


def plot_data_overview(X, labels, responsibility_matrix, centers=None, is_train=True):
    """
    Plots the data with three different visualizations:
    1. Data colored by cluster labels.
    2. Data in a neutral color.
    3. Data colored according to the responsibilities of the clusters.

    Parameters:
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data points to be plotted.

    labels : np.ndarray, shape (n_samples,)
        The cluster labels assigned to each data point.

    responsibility_matrix : np.ndarray, shape (n_clusters, n_samples)
        The responsibility matrix indicating the degree to which each data point belongs to each cluster.

    centers : np.ndarray, shape (n_clusters, n_features), optional
        The mean vectors (centers) of the clusters to be plotted.

    is_train : bool, optional
        If True, indicates that the data is training data. Adjusts the titles accordingly.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_data_colored_by_labels(X, labels, centers, is_train)

    plt.subplot(1, 3, 2)
    plot_data_neutral(X, is_train)

    plt.subplot(1, 3, 3)
    plot_data_colored_by_responsibilities(X, responsibility_matrix, centers, is_train)

    plt.tight_layout()

