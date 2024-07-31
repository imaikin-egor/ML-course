from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
       
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
      
        if self.loss_function == LossFunction.MSE:
            return np.mean(np.square(self.predict(x) - y))
        elif self.loss_function == LossFunction.MAE:
            return np.mean(np.abs(self.predict(x) - y))
        elif self.loss_function == LossFunction.LogCosh:
            x_pred = self.predict(x)
            return np.mean(np.log(np.cosh(x_pred - y)))
        elif self.loss_function == LossFunction.Huber:
            x_pred = self.predict(x)
            delta = 1.0
            loss = np.where(np.abs(x_pred - y) < delta, 0.5 * np.square(x_pred - y), delta * np.abs(x_pred - y) - 0.5 * np.square(delta))
            return np.mean(loss)
        raise NotImplementedError('BaseDescent calc_loss function not implemented')

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.w)


class VanillaGradientDescent(BaseDescent):

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        
        learning_rate = self.lr()
        weight_diff = -learning_rate * gradient
        self.w += weight_diff
        return weight_diff

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        prediction = self.predict(x)
        gradient = (-2/n) * np.dot(x.T, (y - prediction))
        return gradient


class StochasticDescent(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        batch_indices = np.random.randint(n, size=self.batch_size)  # выбор случайных индексов для батчей
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        prediction = self.predict(x_batch)
        gradient = (2/self.batch_size) * np.dot(x_batch.T, (prediction - y_batch))
        return gradient

class MomentumDescent(VanillaGradientDescent):
 
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9
        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        learning_rate = self.lr()
        self.h = self.alpha * self.h + learning_rate * gradient
        weight_diff = -self.h
        self.w += weight_diff
        return weight_diff
        raise NotImplementedError('MomentumDescent update_weights function not implemented')


class Adam(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        eta = self.lr()
        i = self.lr.iteration
        m_next = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v_next = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        self.m = m_next
        self.v = v_next
        m_h_next = m_next / (1 - self.beta_1 ** i)
        v_h_next = v_next / (1 - self.beta_2 ** i)
        w_prev = self.w.copy()
        w_next = w_prev - (eta / (v_h_next ** 0.5 + self.eps)) * m_h_next
        self.w = w_next
        return w_next - w_prev
        raise NotImplementedError('Adagrad update_weights function not implemented')


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu
        
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
         l2_gradient = self.w
         l2_gradient[-1] = 0
         return super().calc_gradient(x, y) + l2_gradient * self.mu
    
class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
