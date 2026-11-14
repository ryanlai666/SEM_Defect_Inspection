"""Few-shot / linear-probe heads on frozen features.

* `LinearProbe`  - multinomial logistic regression (sklearn, L-BFGS) on L2-normalised features,
                   the standard DINOv2 linear-evaluation protocol on [CLS ; mean patch] descriptors.
* `PrototypeClassifier` - nearest class mean with cosine similarity; `center=True` subtracts the
                   support mean before L2 normalisation ("CL2N", SimpleShot, arXiv:1911.04623).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


class LinearProbe:
    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 2000):
        self.clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter)

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearProbe:
        self.clf.fit(l2n(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(l2n(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(l2n(x))


class PrototypeClassifier:
    def __init__(self, center: bool = True):
        self.center = center
        self.mu: np.ndarray | None = None
        self.protos: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _t(self, x: np.ndarray) -> np.ndarray:
        if self.center:
            x = x - self.mu
        return l2n(x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> PrototypeClassifier:
        self.mu = x.mean(0, keepdims=True)
        z = self._t(x)
        self.classes_ = np.unique(y)
        self.protos = l2n(np.stack([z[y == c].mean(0) for c in self.classes_]))
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self._t(x) @ self.protos.T

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.scores(x).argmax(1)]


class NearestNeighbor:
    """Cosine 1-nearest-neighbour on L2-normalised features (training-free reference)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> NearestNeighbor:
        self.x, self.y = l2n(x), np.asarray(y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.y[(l2n(x) @ self.x.T).argmax(1)]
