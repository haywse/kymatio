from sklearn.base import BaseEstimator, TransformerMixin


class ScatteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, S, signal_shape):
        """Creates an object that is compatible with the scikit-learn API
        and implements the `.transform` method.

        Parameters
        ==========

        S: an instance of Scattering1D, Scattering2D or Scattering3D
            This instance is called by the transformer.

        signal_shape: tuple of ints
            The shape of one sample. The `scikit-learn` convention is to work
            with 2D arrays only of shape `(n_samples, n_features)`. Data is
            delivered in this way and has to be reshaped before transforming.

        Output
        ======
        Y: ndarray of shape (n_samples, n_scattering_features)
            The scattering coefficients, raveled in row-major order (C-like).
        """

        self.scattering = S
        self.signal_shape = signal_shape

    def fit(self, X=None, y=None):
        # no fitting necessary
        return self

    def predict(self, X):
        X_reshaped = X.reshape((-1,) + self.signal_shape)

        transformed = self.scattering(X_reshaped)

        output = transformed.reshape(X.shape[0], -1)

        return output

    transform = predict
