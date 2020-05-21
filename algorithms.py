import numpy as np
from abc import abstractmethod


class GradientDecent:
    def __init__(self, A: np.ndarray, b: np.array, x_0: np.array, epsilon: float):
        self.A = A
        self.b = b
        self.x_0 = x_0
        self.epsilon = epsilon  # stopping condition

    def run(self, x, t=0):
        next_x = self.compute_next_x(x, t)
        fx = self.zero_order_oracle(x)
        f_next_x = self.zero_order_oracle(next_x)

        while not self.is_stopping_critiria(fx, f_next_x):
            fx = self.zero_order_oracle(next_x)
            next_x = self.compute_next_x(next_x)
            f_next_x = self.zero_order_oracle(next_x)

        return next_x, f_next_x

    def zero_order_oracle(self, x):
        """
        returns the LR function's value for x
        value is 0.5 * ||Ax - b||^2
        """
        v = np.matmul(self.A, x) - self.b
        v = 0.5 * np.linalg.norm(v, ord=2)**2
        return v

    def first_order_oracle(self, x):
        """
        For linear regression problem we have
        df/dx = A.T (Ax - b)
        """
        gradient = np.matmul(self.A.T, np.matmul(self.A, x) - self.b)
        return gradient

    @abstractmethod
    def step_size(self, *args):
        ...

    @abstractmethod
    def compute_next_x(self, *args):
        ...

    @abstractmethod
    def is_stopping_criteria(self, fx, f_next_x):
        ...


class NonSmoothPGD(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_0: np.array, epsilon: float, R:float):
        super().__init__(A, b, x_0, epsilon)
        

    @property
    def step_size(self, *args):
        return self.R

    def compute_next_x(self, *args):
        ...

    def is_stopping_criteria(self, fx, f_next_x):
        ...

    def calculate_projection(self):
        ...
