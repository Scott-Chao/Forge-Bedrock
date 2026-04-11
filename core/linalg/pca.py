import numpy as np
from .matrix import Matrix
from .decompositions import SVD


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None
        self.explained_variance_ratio = None

    def fit(self, X):
        n_samples = X.rows
        self.mean = np.mean(X.data, axis=0)
        X_centered = X.data - self.mean

        svd = SVD(Matrix(X_centered))

        self.components = Matrix(svd.VT.data[: self.n_components, :])
        full_singular_values = svd.S.data.diagonal()
        self.singular_values = full_singular_values[: self.n_components]

        explained_variance = (self.singular_values**2) / (n_samples - 1)
        total_variance = np.sum(explained_variance)
        explained_variance = explained_variance[: self.n_components]
        self.explained_variance_ratio = explained_variance / total_variance

        return self

    def transform(self, X):
        X_centered = X.data - self.mean
        return Matrix(X_centered @ self.components.data.T)

    def inverse_transform(self, X_reduced):
        return Matrix(X_reduced.data @ self.components.data + self.mean)
