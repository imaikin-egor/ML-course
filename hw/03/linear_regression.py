from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:


    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:       
         self.loss_history.append(self.descent.calc_loss(x, y))
         for i in range(self.max_iter):
          step = self.descent.step(x, y) 
          self.loss_history.append(self.descent.calc_loss(x, y))
          if abs(sum(step ** 2)) < self.tolerance or np.isnan(step).any(): 
            break

        return self
    
    def _compute_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.predict(x)
        gradient = -2 * np.dot(x.T, (y - y_pred)) / y.size

        return gradient

    def _compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self._predict(x)
        loss = np.mean((y_pred - y) ** 2)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:

        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:

        return self.descent.calc_loss(x, y)
