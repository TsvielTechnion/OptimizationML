import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from data_gen import GenData
from algorithms import *

"""
Constant Parameters
"""
sb.set()
columns = 10
rows = 300
x_1 = np.zeros(columns)
epsilon = 0.1

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
dfs_sgd = []
dfs_mb_sgd = []

# sgd = SGD(A, b, x_1, epsilon, R, A_sigma_min)
# sgd_result = sgd.run()
# sgd_result['iterations'] = [i for i in range(len(sgd_result['x']))]
# df_sgd = pd.DataFrame(sgd_result)
# df_sgd["algorithm"] = "Stochastic gradient descent"
# dfs_sgd.append(sgd.weighted_avg(sgd_result['x']))
# print(sgd.weighted_avg(sgd_result['x']))

bs=5
mb_sgd = MiniBatchSGD(A, b, x_1, epsilon, R, A_sigma_min, A_sigma_max, bs)
mb_sgd_result = mb_sgd.run(int(2000/bs))
mb_sgd_result['iterations'] = [i for i in range(len(mb_sgd_result['x']))]
df_mb_sgd = pd.DataFrame(mb_sgd_result)
df_mb_sgd["algorithm"] = "Mini Batch Stochastic gradient descent"
dfs_mb_sgd.append(mb_sgd.weighted_avg(mb_sgd_result['x']))
print(mb_sgd.weighted_avg(mb_sgd_result['x']))




# results = pd.concat([df_smooth_gd, df_non_smooth_pgd, smooth_df_accelerated_gd])
# results = results.loc[:, ~results.columns.duplicated()]

# f, ax = plt.subplots(figsize=(7, 7))
# ax.set(xscale="log", yscale="log")
# sb.lineplot(x=results['iterations'],
#             y=results['fx'],
#             hue=results['algorithm'],
#             ax=ax,
#             )
# plt.show()

