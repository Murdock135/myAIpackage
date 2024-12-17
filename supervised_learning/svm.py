from scipy.io import loadmat
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class SVM:
    def __init__(self, kernel=None, C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None # lagrange multipliers
        self.b = None
        self.support_vectors = None # support vectors
        self.support_labels = None # labels of support vectors
        self.support_alpha = None # lagrange multipliers of support vectors
        self.w = None

    def _gram_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        if self.kernel is None:
            return np.dot(X, X.T)
        else:
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = self.kernel(X[i], X[j])
            return K

    def solve(self, X, y):
        # Get number of samples
        n_samples = X.shape[0]

        # Compute the Gram matrix
        K = self._gram_matrix(X)

        # Define the objective function (dual problem)
        def objective(alpha):
            return 0.5 * np.sum(np.outer(alpha, alpha) * np.outer(y, y) * K) - np.sum(alpha)

        # Equality constraint: sum(alpha_i * y_i) = 0
        linear_constraint = LinearConstraint(y, [0], [0])

        # Bounding box constraint: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Solve the quadratic problem
        initial_alpha = np.zeros(n_samples)
        result = minimize(objective, initial_alpha, bounds=bounds, constraints=[linear_constraint])

        # Check if the solver converged
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Extract the optimized alpha values
        self.alpha = result.x

        # Identify support vectors
        support_vector_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_labels = y[support_vector_indices]
        self.support_alpha = self.alpha[support_vector_indices]

        # Compute the bias term b
        # TODO: vectorize
        b_vals = []
        K_support = K[support_vector_indices]
        for i in range(len(self.support_labels)):
            b = self.support_labels[i] - np.dot(self.alpha * y, K_support[i])
            b_vals.append(b)
        
        self.b = np.mean(b_vals)

        # If it's a linear kernel, compute the weight vector w
        if self.kernel is None:
            self.w = np.dot(self.alpha * y, X)

    def predict(self, X):
        if self.w is not None:
            # Linear case
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            # Non-linear kernel case
            K = np.array([[self.kernel(sv, x) for x in X] for sv in self.support_vectors])

            output_vals = np.dot((self.support_alpha * self.support_labels), K) + self.b

            predictions = np.sign(output_vals)
            return predictions

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=2, gamma=1, coef0=1):
    return (gamma * np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance)

def plot_decision_boundary(svm, X, y, title=''):
    # Plot decision boundary for 2D data
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)