import numpy as np
from abc import abstractmethod


X = "x"
FX = "fx"


class GradientDecent:
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float):
        self.A = A
        self.m = self.A.shape[0]
        self.d = self.A.shape[1]
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

    def run(self, iter=1000):
        # while not self.is_stopping_criteria():
        for i in range(iter):
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

    def first_order_stochastic_oracle(self, x, i=None):
        if i is None:
            i = np.random.random_integers(low=0, high=self.m-1, size=1)
        gradient = np.matmul(np.matmul(self.A[i], x) - self.b[i], self.A[i]) * self.m
        stochastic_gradient = gradient + np.random.normal(loc=0, scale=1, size=self.d)
        return stochastic_gradient

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


class SGD(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, R: float, sigma_min: float):
        self.R = R
        self.sigma_min = sigma_min
        self.alpha = self.sigma_min**2
        super().__init__(A, b, x_1, epsilon)

    def calculate_projection(self, x):
        """
        R * x / (max(||X||, R)
        """
        nominator = self.R * x
        denominator = max(np.linalg.norm(x, 2), self.R)
        return nominator/float(denominator)

    def step_size(self, t):
        return float(2)/(self.alpha*(t+1))

    def compute_next_x(self, x):
        t = len(self.history[X])
        next_y = x - (self.step_size(t) * self.first_order_stochastic_oracle(x))
        next_x = self.calculate_projection(next_y)
        return next_x

    def weighted_avg(self, x):
        weighted_avg_vec = [0]
        for i in range(1, len(x)):
            partial_x = x[:i]
            weighted_avg = np.zeros(self.d)
            for s, x_s in enumerate(partial_x):
                weighted_avg += 2*(s+1)*x_s/float(i*(i+1))
            weighted_avg_vec.append(self.zero_order_oracle(weighted_avg))
        return weighted_avg_vec


class SVGR(GradientDecent):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, sigma_min: float,
                 sigma_max: int, inner_iter: int):
        self.sigma_min = sigma_min
        self.alpha = self.sigma_min**2
        self.beta = sigma_max**2
        self.inner_iter = inner_iter
        self.step_size = self.step_size()
        super().__init__(A, b, x_1, epsilon)

    def step_size(self):
        return float(1)/(10*self.beta)

    def compute_next_x(self, x):
        average_gradeient = self.calc_avg_gradient(x)
        next_inner_x = x
        for t in range(self.inner_iter):
            i = np.random.random_integers(low=0, high=self.m-1, size=1)
            update = self.step_size * (self.first_order_stochastic_oracle(next_inner_x, i) -
                                       self.first_order_stochastic_oracle(x, i) +
                                       average_gradeient)
            next_inner_x = np.subtract(next_inner_x, update)
        return next_inner_x

    def calc_avg_gradient(self, x):
        average_gradient = np.zeros(self.d)
        for i in range(self.m):
            average_gradient = np.add(average_gradient, self.first_order_stochastic_oracle(x))
        average_gradient /= self.m
        return average_gradient


class MiniBatchSGD(SGD):
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, R: float, sigma_min: float,
                 sigma_max: int, bs: int):
        self.bs = bs
        self.beta = sigma_max**2
        self.A_sigma_max = sigma_max
        super().__init__(A, b, x_1, epsilon, R, sigma_min)

    def compute_next_x(self, x):
        t = len(self.history[X])
        fo_stochastic_oracles = np.zeros(self.d)
        for i in range(self.bs):
            fo_stochastic_oracles = np.add(fo_stochastic_oracles, self.first_order_stochastic_oracle(x))
        next_y = x - ((self.step_size(t)/float(self.bs)) * fo_stochastic_oracles)
        next_x = self.calculate_projection(next_y)
        return next_x

    def weighted_avg(self, x):
        weighted_avg_vec = [0]
        for i in range(1, len(x)):
            partial_x = x[:i]
            weighted_avg = np.zeros(self.d)
            for s, x_s in enumerate(partial_x):
                weighted_avg += x_s/float(i)
            weighted_avg_vec.append(self.zero_order_oracle(weighted_avg))
        return weighted_avg_vec

    def step_size(self, t):
        eta = self.calc_eta(t)
        return float(1) / (self.beta + 1 / (eta))

    def calc_eta(self, t):
        sigma = self.calc_sigma()
        eta = ((self.R / sigma) * (np.sqrt(2 / t)))
        return eta

    def calc_sigma(self):
        B = np.sqrt(self.calculate_lipschitz())
        sigma = np.sqrt((2 * B**2) / self.bs)
        return sigma

    def calculate_lipschitz(self):
        """
        Calculate L-Lipschitz parameter of LR function by: sigma_max(A)^2*R + sigma_max(A)*||b||
        """
        first_summand = self.A_sigma_max ** 2 * self.R
        second_summand = self.A_sigma_max * np.linalg.norm(self.b, ord=2)
        return first_summand + second_summand


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


class SmoothAcceleratedGD(GradientDecent):
    """
    For Smooth and Non Strongly convex
    """
    def __init__(self, A: np.ndarray, b: np.array, x_1: np.array, epsilon: float, beta: float):
        self.beta = beta
        self.y = x_1
        self.prev_lambda = 0
        self.current_lambda = self.compute_lambda(self.prev_lambda)
        self.next_lambda = self.compute_lambda(self.current_lambda)
        super().__init__(A, b, x_1, epsilon)

    def step_size(self):
        return 1 / self.beta

    def compute_next_x(self, x):
        gradient = self.first_order_oracle(x)
        next_y = x - (self.step_size() * gradient)

        gamma = self.compute_gamma()
        next_x = (1 - gamma) * next_y + gamma * self.y

        self.y = next_y
        self.update_lambda()

        return next_x

    def compute_gamma(self):
        """
        Compute value of gamma in current iteration by: gamma_t = (1-lambda_t)/lambda_t+1
        """
        return (1 - self.current_lambda) / self.next_lambda

    def compute_lambda(self, prev_lambda):
        """
        Compute the value of lambda in current iteration by: (1+sqrt(1+4*prev_lambda^2))/2
        :param prev_lambda:
        :return:
        """
        return 0.5 * (1 + np.sqrt(1 + 4 * prev_lambda ** 2))

    def update_lambda(self):
        self.prev_lambda = self.current_lambda
        self.current_lambda = self.next_lambda
        self.next_lambda = self.compute_lambda(self.current_lambda)