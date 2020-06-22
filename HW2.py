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
dfs_mb_sgd_5 = []
dfs_mb_sgd_10 = []
dfs_mb_sgd_20 = []
dfs_svgr = []

wa = pd.DataFrame()

for i in range(2):
    sgd = SGD(A, b, x_1, epsilon, R, A_sigma_min)
    sgd_result = sgd.run(2000)
    sgd_result['iterations'] = [i for i in range(len(sgd_result['x']))]
    df_sgd = pd.DataFrame(sgd_result)
    df_sgd["algorithm"] = "SGD"
    sdg_wa = pd.DataFrame(data=sgd.weighted_avg(sgd_result['x']), columns=["SGD"])
    wa = pd.concat([wa, sdg_wa], axis=1)
    dfs_sgd.append(df_sgd)

    bs=5
    mb_sgd_5 = MiniBatchSGD(A, b, x_1, epsilon, R, A_sigma_min, A_sigma_max, bs)
    mb_sgd_5_result = mb_sgd_5.run(int(2000/bs))
    mb_sgd_5_result['iterations'] = [i for i in range(len(mb_sgd_5_result['x']))]
    df_mb_sgd_5 = pd.DataFrame(mb_sgd_5_result)
    df_mb_sgd_5["algorithm"] = "MB_SGD_5"
    mb_sgd_5_wa = pd.DataFrame(data=mb_sgd_5.weighted_avg(mb_sgd_5_result['x']), columns=["MiniBatchSGD_5"])
    wa = pd.concat([wa, mb_sgd_5_wa], axis=1)
    dfs_mb_sgd_5.append(df_mb_sgd_5)

    bs = 10
    mb_sgd_10 = MiniBatchSGD(A, b, x_1, epsilon, R, A_sigma_min, A_sigma_max, bs)
    mb_sgd_10_result = mb_sgd_10.run(int(2000 / bs))
    mb_sgd_10_result['iterations'] = [i for i in range(len(mb_sgd_10_result['x']))]
    df_mb_sgd_10 = pd.DataFrame(mb_sgd_10_result)
    df_mb_sgd_10["algorithm"] = "MB_SGD_10"
    mb_sgd_10_wa = pd.DataFrame(data=mb_sgd_10.weighted_avg(mb_sgd_10_result['x']), columns=["MiniBatchSGD_10"])
    wa = pd.concat([wa, mb_sgd_10_wa], axis=1)
    dfs_mb_sgd_10.append(df_mb_sgd_10)

    bs = 20
    mb_sgd_20 = MiniBatchSGD(A, b, x_1, epsilon, R, A_sigma_min, A_sigma_max, bs)
    mb_sgd_20_result = mb_sgd_20.run(int(2000 / bs))
    mb_sgd_20_result['iterations'] = [i for i in range(len(mb_sgd_20_result['x']))]
    df_mb_sgd_20 = pd.DataFrame(mb_sgd_20_result)
    df_mb_sgd_20["algorithm"] = "MB_SGD_20"
    mb_sgd_20_wa = pd.DataFrame(data=mb_sgd_20.weighted_avg(mb_sgd_20_result['x']), columns=["MiniBatchSGD_20"])
    wa = pd.concat([wa, mb_sgd_20_wa], axis=1)
    dfs_mb_sgd_20.append(df_mb_sgd_20)

    inner_iter = 10
    svgr = SVGR(A, b, x_1, epsilon, A_sigma_min, A_sigma_max, inner_iter)
    svgr_result = svgr.run(2000)
    svgr_result['iterations'] = [i for i in range(len(svgr_result['x']))]
    df_svgr = pd.DataFrame(svgr_result)
    df_svgr["algorithm"] = "SVGR"
    svgr_wa = pd.DataFrame(data=svgr_result['fx'], columns=["SVGR"])
    wa = pd.concat([wa, svgr_wa], axis=1)
    dfs_svgr.append(df_svgr)


df_sgd = pd.concat(dfs_sgd, axis=1)
df_sgd["fx"] = df_sgd['fx'].mean(axis=1)

df_mb_sgd_5 = pd.concat(dfs_mb_sgd_5, axis=1)
df_mb_sgd_5["fx"] = df_mb_sgd_5['fx'].mean(axis=1)

df_mb_sgd_10 = pd.concat(dfs_mb_sgd_10, axis=1)
df_mb_sgd_10["fx"] = df_mb_sgd_10['fx'].mean(axis=1)

df_mb_sgd_20 = pd.concat(dfs_mb_sgd_20, axis=1)
df_mb_sgd_20["fx"] = df_mb_sgd_20['fx'].mean(axis=1)

df_svgr = pd.concat(dfs_svgr, axis=1)
df_svgr["fx"] = df_svgr['fx'].mean(axis=1)

results = pd.concat([df_sgd, df_mb_sgd_5, df_mb_sgd_10, df_mb_sgd_20, df_svgr])
results = results.loc[:, ~results.columns.duplicated()]

f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log", yscale="log")
sb.lineplot(x=results['iterations'],
            y=results['fx'],
            hue=results['algorithm'],
            ax=ax,
            )
plt.show()

wa.plot()
plt.show()

