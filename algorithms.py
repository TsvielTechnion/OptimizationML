import numpy as np
from abc import abstractmethod


X = "x"
FX = "fx"


class GradientDecent:
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float):
        self.A = A
        self.b = b
        self.x_1 = x_1
        self.epsilon = epsilon  # stopping condition

        self.history = {
                        X: [x_1],
                        FX: [self.zero_order_oracle(x_1)]
                    }  # [[x values], [fx values]]

        x_2 = self.compute_next_x(x_1)
        f_x_2 = self.zero_order_oracle(x_2)
        self.history[X].append(x_2)
        self.history[FX].append(f_x_2)

    def run(self):
        while not self.is_stopping_criteria():
            next_x = self.compute_next_x(self.last_x)
            f_next_x = self.zero_order_oracle(next_x)
            self.history[X].append(next_x)
            self.history[FX].append(f_next_x)

        return self.history

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

    def is_stopping_criteria(self):
        fx, f_next_x = self.history[FX][-2:]
        should_stop = True if fx - f_next_x <= self.epsilon else False
        return should_stop

    @abstractmethod
    def step_size(self, *args):
        ...

    @abstractmethod
    def compute_next_x(self, *args):
        ...

    @property
    def last_x(self):
        return self.history[X][-1]


class NonSmoothPGD(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, R:float, A_sigma_max: float):
        self.R = R
        self.A_sigma_max = A_sigma_max
        self.b = b
        self.L = self.calculate_lipschitz()
        super().__init__(A, b, x_1, epsilon)
        self.f_avg_history = [self.zero_order_oracle(x_1)]

    def calculate_lipschitz(self):
        """
        Calculate L-Lipschitz parameter of LR function by: sigma_max(A)^2*R + sigma_max(A)*||b||
        """
        first_summand = self.A_sigma_max ** 2 * self.R
        second_summand = self.A_sigma_max * np.linalg.norm(self.b, ord=2)
        return first_summand + second_summand

    def step_size(self, t):
        return self.R / (self.L * np.sqrt(t))

    def compute_next_x(self, x):
        t = len(self.history[X])
        next_y = x - (self.step_size(t) * self.first_order_oracle(x))
        next_x = self.calculate_projection(next_y)
        return next_x

    def is_stopping_criteria(self):
        fx = self.f_avg_history[-1]
        f_next_x = self.calculate_mean()

        if fx - f_next_x <= self.epsilon:
            # The relevant history is of the AVG
            self.history[FX] = self.f_avg_history
            return True
        return False

    def calculate_mean(self):
        x_avg = np.mean(self.history[X], axis=0)
        f_x_avg = self.zero_order_oracle(x_avg)
        self.f_avg_history.append(f_x_avg)
        return f_x_avg

    def calculate_projection(self, x):
        """
        R * x / (max(||X||, R)
        """
        nominator = self.R * x
        denominator = max(np.linalg.norm(x, 2), self.R)
        return nominator/float(denominator)


class SmoothGD(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, beta: float):
        self.beta = beta
        super().__init__(A, b, x_1, epsilon)

    def step_size(self):
        return 1/self.beta

    def compute_next_x(self, x):
        gradient = self.first_order_oracle(x)
        next_x = x - self.step_size() * gradient
        return next_x


class AcceleratedGD(GradientDecent):
    """
    For Strongly convex
    """
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, beta: float, alpha: float):
        self.alpha = alpha
        self.beta = beta
        self.k = self.beta / self.alpha
        self.y = x_1
        super().__init__(A, b, x_1, epsilon)

    def step_size(self):
        return 1/self.beta

    def compute_next_x(self, x):
        gradient = self.first_order_oracle(x)
        next_y = x - self.step_size() * gradient
        
        sqrt_k = np.sqrt(self.k)
        w = (sqrt_k - 1) / (sqrt_k + 1)
        lhs = (1 + w) * next_y
        rhs = w * self.y

        next_x = lhs - rhs
        self.y = next_y

        return next_x
