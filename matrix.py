import numpy as np


class Const:
    SIGMA_LOW = 1
    SIGMA_MAX = 10
    COLUMNS = 10
    ROWS = 20
    OPTIMAL_SOLUTION = np.array([1] * COLUMNS)


class GenData:
    def __init__(self, sigma_min: int,
                 sigma_max: int,
                 columns: int,
                 rows: int,
                 optimal_sol: np.array):
        assert rows >= columns, "Rows must be greater or equal to columns"
        assert sigma_min >= 0, "Singular values must be geq than 0"
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.columns = columns
        self.rows = rows
        self.optimal_solution = optimal_sol

    def gen_matrix(self):
        """
        Generating the matrix using SVD decomposition:
        Creating orthogonal matrices and Q, P and diagonal D.
        :return:
        """
        P = self.create_random_orthogonal_matrix(self.rows)
        Q = self.create_random_orthogonal_matrix(self.columns)
        D = self.create_diagonal_matrix()

        matrix = np.matmul(np.matmul(P, D), Q)

        return matrix

    @staticmethod
    def create_random_orthogonal_matrix(dim=3):
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())
            # Householder transformation
            Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
            mat = np.eye(dim)
            mat[n - 1:, n - 1:] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H

    @staticmethod
    def is_pos_def(matrix):
        return bool(np.all(np.linalg.eigvals(matrix) > 0))

    def create_diagonal_matrix(self):
        """
        if sigma low > 0 you will get matrix A which
        satisfies transpose(A) * A is a positive definite matrix
        :return:
        """
        diag = np.random.randint(low=self.sigma_min, high=self.sigma_max, size=self.columns)
        diag = np.sort(diag)[::-1]

        diag[0] = self.sigma_max
        diag[-1] = self.sigma_min
        D = np.diag(diag)
        lower_pad = np.zeros(shape=(self.rows - self.columns, self.columns))
        return np.concatenate((D, lower_pad), axis=0)

    def create_b(self, matrix):
        noise = np.random.normal(loc=0, scale=0.001, size=self.rows)
        b = np.matmul(matrix, self.optimal_solution) + noise
        return b

    def gen_data(self):
        matrix = self.gen_matrix()
        b = self.create_b(matrix)
        return matrix, b


if __name__ == "__init__":

    optimal_solution = Const.OPTIMAL_SOLUTION
    gen = GenData(Const.SIGMA_LOW,
                  Const.SIGMA_MAX,
                  Const.COLUMNS,
                  Const.ROWS,
                  optimal_solution)

    R = np.linalg.norm(optimal_solution, ord=2)
