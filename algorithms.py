import numpy as np
from abc import abstractmethod
from copy import deepcopy


class GradientDecent:
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float):
        self.A = A
        self.b = b
        self.x_1 = x_1
        self.epsilon = epsilon  # stopping condition

    def run(self, x, t=1):
        next_x = self.compute_next_x(x, t)
        fx = self.zero_order_oracle(x)
        f_next_x = self.zero_order_oracle(next_x)

        while not self.is_stopping_criteria(fx, f_next_x):
            fx = f_next_x
            t += 1
            next_x = self.compute_next_x(next_x, t)
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
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, R:float, A_sigma_max: float):
        super().__init__(A, b, x_1, epsilon)
        self.R = R
        self.A_sigma_max = A_sigma_max
        self.L = self.calculate_lipschitz()
        self.x_values = [x_1]

    def calculate_lipschitz(self):
        """
        Calculate L-Lipschitz parameter of LR function by: sigma_max(A)^2*R + sigma_max(A)*||b||
        :return:
        """
        return self.A_sigma_max**2*self.R + self.A_sigma_max*np.linalg.norm(self.b, ord=2)

    @property
    def step_size(self, *args):
        t = args[0]
        return self.R/(self.L*np.sqrt(t))

    def compute_next_x(self, *args):
        x = args[0]
        t = args[1]

        next_y = x - self.step_size(t)*self.first_order_oracle(x)
        next_x = self.calculate_projection(next_y)
        self.x_values.append(next_x)
        return next_x

    def is_stopping_criteria(self, fx, f_next_x):
        x_values_t_minus_1 = deepcopy(self.x_values).pop()

        x_values_t_minus_1_avg = np.mean(x_values_t_minus_1)
        x_values_avg = np.mean(self.x_values)

        fx = self.zero_order_oracle(x_values_t_minus_1_avg)
        f_next_x = self.zero_order_oracle(x_values_avg)

        if f_next_x - fx <= self.epsilon:
            return True
        else:
            return False

    def calculate_projection(self, x):
        nominator = self.R * x
        denominator = np.amax((np.linalg.norm(x, 2), self.R))
        return nominator/float(denominator)


class SmoothGD(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, beta: float):
        super().__init__(A, b, x_1, epsilon)
        self.beta = beta

    def step_size(self, *args):
        return 1/self.beta

    def compute_next_x(self, *args):
        x = args[0]
        gradient = self.first_order_oracle(x)
        next_x = x - self.step_size()*gradient
        return next_x

    def is_stopping_criteria(self, fx, f_next_x):
        if f_next_x - fx < self.epsilon:
            return True
        else:
            return False


class AcceleratedGD(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, alpha: float, beta: float):
        super().__init__(A, b, x_1, epsilon)
        self.alpha = alpha
        self.beta = beta
        self.k = self.beta/self.alpha
        self.y = x_1

    def step_size(self, *args):
        return 1/self.beta

    def compute_next_x(self, *args):
        x = args[0]
        gradient = self.first_order_oracle(x)
        next_y = x - self.step_size()*gradient
        
        sqrt_k = np.sqrt(self.k)
        lhs = (1 + (sqrt_k  - 1)/(sqrt_k + 1))*next_y
        rhs = ((sqrt_k - 1)/(sqrt_k + 1))*self.y

        next_x = lhs - rhs
        self.y = next_y

        return next_x

    def is_stopping_criteria(self, fx, f_next_x):
        if f_next_x - fx < self.epsilon:
            return True
        else:
            return False