import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from matrix import GenData
from algorithms import NonSmoothPGD, SmoothGD, AcceleratedGD


"""
    Starting With Positive Definite A.T * A
"""
A_sigma_min = 0
A_sigma_max = 5
columns = 10
rows = 100

optimal_solution = np.random.randint(low=1, high=5, size=10)
R = np.linalg.norm(optimal_solution, ord=2)
gen = GenData(sigma_min=A_sigma_min, sigma_max=A_sigma_max, columns=columns, rows=rows, optimal_sol=optimal_solution)
A, b = gen.gen_data()

x_1 = np.zeros(columns)
epsilon = 0.01

"""
Running the algorithm
"""
# non_smooth_pgd = NonSmoothPGD(A, b, x_1, epsilon, R, A_sigma_max)
# non_smooth_pgd_result = non_smooth_pgd.run()

# smooth_gd = SmoothGD(A, b, x_1, epsilon, A_sigma_max**2)
# smooth_gd_result = smooth_gd.run()

accelerated_gd = AcceleratedGD(A, b, x_1, epsilon, A_sigma_max**2, A_sigma_min**2)
accelerated_gd_result = accelerated_gd.run()

sb.lineplot(x=[i for i in range(len(accelerated_gd_result['x']))],
            y=accelerated_gd_result['fx'])
plt.show()
