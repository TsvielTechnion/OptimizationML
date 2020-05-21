import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from matrix import GenData
from algorithms import NonSmoothPGD, SmoothGD, AcceleratedGD

"""
Constant Parameters
"""
sb.set()
columns = 10
rows = 100
x_1 = np.zeros(columns)
epsilon = 0.01

"""
    Starting With Positive Definite A.T * A
"""
A_sigma_min = 1
A_sigma_max = 10


optimal_solution = np.random.randint(low=A_sigma_min, high=A_sigma_max, size=columns)
R = np.linalg.norm(optimal_solution, ord=2)
gen = GenData(sigma_min=A_sigma_min, sigma_max=A_sigma_max, columns=columns, rows=rows, optimal_sol=optimal_solution)
A, b = gen.gen_data()

"""
Running the algorithms
"""

non_smooth_pgd = NonSmoothPGD(A, b, x_1, epsilon, R, A_sigma_max)
non_smooth_pgd_result = non_smooth_pgd.run()
non_smooth_pgd_result['iterations'] = [i for i in range(len(non_smooth_pgd_result['x']))]
df_non_smooth_pgd = pd.DataFrame(non_smooth_pgd_result)
df_non_smooth_pgd["algorithm"] = "Non Smooth PGD"

smooth_gd = SmoothGD(A, b, x_1, epsilon, A_sigma_max**2)
smooth_gd_result = smooth_gd.run()
smooth_gd_result['iterations'] = [i for i in range(len(smooth_gd_result['x']))]
df_smooth_gd = pd.DataFrame(smooth_gd_result)
df_smooth_gd["algorithm"] = "Smooth GD"


accelerated_gd = AcceleratedGD(A, b, x_1, epsilon, A_sigma_max**2, A_sigma_min**2)
accelerated_gd_result = accelerated_gd.run()
accelerated_gd_result['iterations'] = [i for i in range(len(accelerated_gd_result['x']))]
df_accelerated_gd = pd.DataFrame(accelerated_gd_result)
df_accelerated_gd["algorithm"] = "Accelerated GD"

results = pd.concat([df_smooth_gd, df_non_smooth_pgd, df_accelerated_gd])

f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log", yscale="log")
sb.lineplot(x=results['iterations'],
            y=results['fx'],
            hue=results['algorithm'],
            ax=ax,
            )
plt.show()

"""
Not Positive Definite A.T * A
"""
# A_sigma_min = 0
# A_sigma_max = 5
#
# optimal_solution = np.random.randint(low=A_sigma_min, high=A_sigma_max, size=columns)
# R = np.linalg.norm(optimal_solution, ord=2)
# gen = GenData(sigma_min=A_sigma_min, sigma_max=A_sigma_max, columns=columns, rows=rows, optimal_sol=optimal_solution)
# A, b = gen.gen_data()


"""
Running the algorithms
"""
# non_smooth_pgd = NonSmoothPGD(A, b, x_1, epsilon, R, A_sigma_max)
# non_smooth_pgd_result = non_smooth_pgd.run()
# non_smooth_pgd_result['iterations'] = [i for i in range(len(non_smooth_pgd_result['x']))]
# df_non_smooth_pgd = pd.DataFrame(non_smooth_pgd_result)
# df_non_smooth_pgd["algorithm"] = "Non Smooth PGD"
#
# smooth_gd = SmoothGD(A, b, x_1, epsilon, A_sigma_max**2)
# smooth_gd_result = smooth_gd.run()
# smooth_gd_result['iterations'] = [i for i in range(len(smooth_gd_result['x']))]
# df_smooth_gd = pd.DataFrame(smooth_gd_result)
# df_smooth_gd["algorithm"] = "Smooth GD"
#
#
# accelerated_gd = AcceleratedGD(A, b, x_1, epsilon, A_sigma_max**2, A_sigma_min**2)
# accelerated_gd_result = accelerated_gd.run()
# accelerated_gd_result['iterations'] = [i for i in range(len(accelerated_gd_result['x']))]
# df_accelerated_gd = pd.DataFrame(accelerated_gd_result)
# df_accelerated_gd["algorithm"] = "Accelerated GD"
#
# results = pd.concat([df_smooth_gd, df_non_smooth_pgd, df_accelerated_gd])
#
# f, ax = plt.subplots(figsize=(7, 7))
# ax.set(xscale="log", yscale="log")
# sb.lineplot(x=results['iterations'],
#             y=results['fx'],
#             hue=results['algorithm'],
#             ax=ax,
#             )
# plt.show()
