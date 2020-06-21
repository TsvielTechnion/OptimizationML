import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from data_gen import GenData
from algorithms import NonSmoothPGD, SmoothGD, AcceleratedGD, SmoothAcceleratedGD

"""
Constant Parameters
"""
sb.set()
columns = 10
rows = 50
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
dfs_non_smooths = []
dfs_smooth = []
dfs_accelarated = []

for i in range(50):
    non_smooth_pgd = NonSmoothPGD(A, b, x_1, epsilon, R, A_sigma_max)
    non_smooth_pgd_result = non_smooth_pgd.run()
    non_smooth_pgd_result['iterations'] = [i for i in range(len(non_smooth_pgd_result['x']))]
    df_non_smooth_pgd = pd.DataFrame(non_smooth_pgd_result)
    df_non_smooth_pgd["algorithm"] = "Non Smooth PGD"
    dfs_non_smooths.append(df_non_smooth_pgd)

    smooth_gd = SmoothGD(A, b, x_1, epsilon, A_sigma_max**2)
    smooth_gd_result = smooth_gd.run()
    smooth_gd_result['iterations'] = [i for i in range(len(smooth_gd_result['x']))]
    df_smooth_gd = pd.DataFrame(smooth_gd_result)
    df_smooth_gd["algorithm"] = "Smooth GD"
    dfs_smooth.append(df_smooth_gd)

    accelerated_gd = AcceleratedGD(A, b, x_1, epsilon, A_sigma_max**2, A_sigma_min**2)
    accelerated_gd_result = accelerated_gd.run()
    accelerated_gd_result['iterations'] = [i for i in range(len(accelerated_gd_result['x']))]
    df_accelerated_gd = pd.DataFrame(accelerated_gd_result)
    df_accelerated_gd["algorithm"] = "Accelerated GD"
    dfs_accelarated.append(df_accelerated_gd)

df_smooth_gd = pd.concat(dfs_smooth, axis=1)
df_smooth_gd["fx"] = df_smooth_gd['fx'].mean(axis=1)

df_non_smooth_pgd = pd.concat(dfs_non_smooths, axis=1)
df_non_smooth_pgd["fx"] = df_non_smooth_pgd['fx'].mean(axis=1)

smooth_df_accelerated_gd = pd.concat(dfs_accelarated, axis=1)
smooth_df_accelerated_gd["fx"] = smooth_df_accelerated_gd['fx'].mean(axis=1)

results = pd.concat([df_smooth_gd, df_non_smooth_pgd, smooth_df_accelerated_gd])
results = results.loc[:, ~results.columns.duplicated()]

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
A_sigma_min = 0
A_sigma_max = 10


"""
Running the algorithms
"""
dfs_non_smooths = []
dfs_smooth = []
dfs_accelarated = []

for i in range(50):
    optimal_solution = np.random.randint(low=A_sigma_min, high=A_sigma_max, size=columns)
    R = np.linalg.norm(optimal_solution, ord=2)
    gen = GenData(sigma_min=A_sigma_min, sigma_max=A_sigma_max, columns=columns, rows=rows, optimal_sol=optimal_solution)
    A, b = gen.gen_data()

    non_smooth_pgd = NonSmoothPGD(A, b, x_1, epsilon, R, A_sigma_max)
    non_smooth_pgd_result = non_smooth_pgd.run()
    non_smooth_pgd_result['iterations'] = [i for i in range(len(non_smooth_pgd_result['x']))]
    df_non_smooth_pgd = pd.DataFrame(non_smooth_pgd_result)
    df_non_smooth_pgd["algorithm"] = "Non Smooth PGD"
    dfs_non_smooths.append(df_non_smooth_pgd)

    smooth_gd = SmoothGD(A, b, x_1, epsilon, A_sigma_max**2)
    smooth_gd_result = smooth_gd.run()
    smooth_gd_result['iterations'] = [i for i in range(len(smooth_gd_result['x']))]
    df_smooth_gd = pd.DataFrame(smooth_gd_result)
    df_smooth_gd["algorithm"] = "Smooth GD"
    dfs_smooth.append(df_smooth_gd)

    smooth_accelerated_gd = SmoothAcceleratedGD(A, b, x_1, epsilon, A_sigma_max**2)
    smooth_accelerated_gd_result = smooth_accelerated_gd.run()
    smooth_accelerated_gd_result['iterations'] = [i for i in range(len(smooth_accelerated_gd_result['x']))]
    smooth_df_accelerated_gd = pd.DataFrame(smooth_accelerated_gd_result)
    smooth_df_accelerated_gd["algorithm"] = "Accelerated GD"
    dfs_accelarated.append(smooth_df_accelerated_gd)

df_smooth_gd = pd.concat(dfs_smooth, axis=1)
df_smooth_gd["fx"] = df_smooth_gd['fx'].mean(axis=1)

df_non_smooth_pgd = pd.concat(dfs_non_smooths, axis=1)
df_non_smooth_pgd["fx"] = df_non_smooth_pgd['fx'].mean(axis=1)

smooth_df_accelerated_gd = pd.concat(dfs_accelarated, axis=1)
smooth_df_accelerated_gd["fx"] = smooth_df_accelerated_gd['fx'].mean(axis=1)

results = pd.concat([df_smooth_gd, df_non_smooth_pgd, smooth_df_accelerated_gd])
results = results.loc[:, ~results.columns.duplicated()]

f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log", yscale="log")
sb.lineplot(x=results['iterations'],
            y=results['fx'],
            hue=results['algorithm'],
            ax=ax,
            )
plt.show()
