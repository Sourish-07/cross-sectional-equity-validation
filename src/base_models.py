# base_models.py
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class LinearSGDModel:
    """
    Scalable logistic regression with:
    - class imbalance handling
    - proper partial_fit semantics
    """

    def __init__(self):
        self.scaler = StandardScaler(with_mean=False)
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1000,
            learning_rate="adaptive",
            eta0=0.01,
            warm_start=True,
            shuffle=False,
            random_state=42,
        )
        self.initialized = False

    def fit(self, X, y, chunk_size=500_000):
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        sample_weight = np.where(y == 1, pos_weight, 1.0)

        for i in range(0, len(X), chunk_size):
            Xc = X[i:i + chunk_size]
            yc = y[i:i + chunk_size]
            wc = sample_weight[i:i + chunk_size]

            Xc = self.scaler.partial_fit(Xc).transform(Xc)

            if not self.initialized:
                self.model.partial_fit(
                    Xc, yc, classes=np.array([0, 1]), sample_weight=wc
                )
                self.initialized = True
            else:
                self.model.partial_fit(Xc, yc, sample_weight=wc)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]
