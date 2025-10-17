import numpy as np
import os
import inspect
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--num_sim", default=10, type=int)
parser.add_argument("--T", default=10000000, type=int)
parser.add_argument("--order_alpha", default=1/12, type=float)

args = parser.parse_args()

num_sim = args.num_sim

time_base = 1 + np.arange(args.T)

# Define the number of points you want to extract
num_points = 100

# Generate logarithmically spaced indices
log_indices = np.logspace(0, np.log2(args.T - 1), num=num_points, base=2)

# Round to the nearest integer and convert to int
log_indices = np.unique(np.round(log_indices).astype(int))  # Ensure unique indices

time_base = time_base[log_indices]

FIGURE_PATH_REGRET = os.path.join(CUR_DIR, "results_multi",
                                  "profit_regret_alpha_" + "{:.4f}".format(args.order_alpha) + ".pdf")
FIGURE_PATH_AVG_QUEUE = os.path.join(CUR_DIR, "results_multi",
                                     "avg_queue_alpha_" + "{:.4f}".format(args.order_alpha) + ".pdf")
FIGURE_PATH_MAX_QUEUE = os.path.join(CUR_DIR, "results_multi",
                                     "max_queue_alpha_" + "{:.4f}".format(args.order_alpha) + ".pdf")

'''
known_two_price (Varma), learn_threshold (Yang), learn_two_price_threshold (proposed),
'''
policy_list = ["known_two_price", "learn_threshold", "learn_two_price_threshold"]
algorithm_list = ["Two-price policy - known\n(Varma et al., 2023)", "Threshold policy\n(Yang & Ying, 2024)", "Probabilistic two-price policy\n(Proposed)"]
order_alpha = [args.order_alpha, args.order_alpha, args.order_alpha]
pd_data_regret_list = []
pd_data_avg_queue_list = []
pd_data_max_queue_list = []
for i in range(len(policy_list)):
    policy = policy_list[i]
    PATH_PROFIT = os.path.join(CUR_DIR,
                               "data_multi/profit_"
                               + policy
                               + "_alpha_"
                               + "{:.4f}".format(order_alpha[i])
                               + "_T_" + str(args.T)
                               + ".npy")
    PATH_AVG_QUEUE = os.path.join(CUR_DIR,
                                  "data_multi/avg_queue_"
                                  + policy
                                  + "_alpha_"
                                  + "{:.4f}".format(order_alpha[i])
                                  + "_T_" + str(args.T)
                                  + ".npy")
    PATH_MAX_QUEUE = os.path.join(CUR_DIR,
                                  "data_multi/max_queue_"
                                  + policy
                                  + "_alpha_"
                                  + "{:.4f}".format(order_alpha[i])
                                  + "_T_" + str(args.T)
                                  + ".npy")
    profit_regret = np.load(PATH_PROFIT)
    avg_queue = np.load(PATH_AVG_QUEUE)
    max_queue = np.load(PATH_MAX_QUEUE)

    profit_regret = profit_regret[log_indices, 0:num_sim]
    avg_queue = avg_queue[log_indices, 0:num_sim]
    max_queue = max_queue[log_indices, 0:num_sim]

    pd_data_regret = pd.DataFrame(np.concatenate((time_base.repeat(num_sim).reshape((-1, 1)),
                                                  profit_regret.reshape((-1, 1))), axis=1),
                                  columns=["Time", "Profit Regret"])
    pd_data_avg_queue = pd.DataFrame(np.concatenate((time_base.repeat(num_sim).reshape((-1, 1)),
                                                     avg_queue.reshape((-1, 1))), axis=1),
                                     columns=["Time", "Average Queue Length"])
    pd_data_max_queue = pd.DataFrame(np.concatenate((time_base.repeat(num_sim).reshape((-1, 1)),
                                                     max_queue.reshape((-1, 1))), axis=1),
                                     columns=["Time", "Maximum Queue Length"])

    pd_data_regret["Algorithm"] = algorithm_list[i]
    pd_data_avg_queue["Algorithm"] = algorithm_list[i]
    pd_data_max_queue["Algorithm"] = algorithm_list[i]

    pd_data_regret_list.append(pd_data_regret)
    pd_data_avg_queue_list.append(pd_data_avg_queue)
    pd_data_max_queue_list.append(pd_data_max_queue)

pd_data_regret_total = pd.concat(pd_data_regret_list, axis=0, ignore_index=True)
pd_data_avg_queue_total = pd.concat(pd_data_avg_queue_list, axis=0, ignore_index=True)
pd_data_max_queue_total = pd.concat(pd_data_max_queue_list, axis=0, ignore_index=True)


plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Profit Regret", markers=False, data=pd_data_regret_total, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="Algorithm", style="Algorithm",
                      height=4,
                      aspect=1.3
                      # legend=False
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.14, 0.92), title=None, frameon=True)
    fig.set_axis_labels('Time', 'Profit Regret', labelpad=10, fontsize=10)
    fig.ax.set_xscale("log", base=2)
    fig.ax.set_yscale("log", base=2)

    fig.set(ylim=(2 ** 0, 2 ** 28))
    fig.set(xlim=(2 ** 1, 2 ** 23))
    plt.xticks([2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608],
               [f"$2^{{{i * 2 - 1}}}$" for i in range(1, 13)])

    plt.grid()

    fig.savefig(FIGURE_PATH_REGRET)
    # plt.show()

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Average Queue Length", markers=False, data=pd_data_avg_queue_total, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="Algorithm", style="Algorithm",
                      height=4,
                      aspect=1.3
                      # legend=False
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.285, 0.50), title=None, frameon=True)
    fig.set_axis_labels('Time', 'Average Queue Length', labelpad=10, fontsize=10)
    fig.ax.set_xscale("log", base=2)
    fig.ax.set_yscale("log", base=2)

    fig.set(ylim=(2 ** (-2), 2 ** 5))
    fig.set(xlim=(2 ** 1, 2 ** 23))
    plt.xticks([2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608],
               [f"$2^{{{i * 2 - 1}}}$" for i in range(1, 13)])

    plt.grid()

    fig.savefig(FIGURE_PATH_AVG_QUEUE)
    # plt.show()

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Maximum Queue Length", markers=False, data=pd_data_max_queue_total, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="Algorithm", style="Algorithm",
                      height=4,
                      aspect=1.3
                      # legend=False
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.285, 0.50), title=None, frameon=True)
    fig.set_axis_labels('Time', 'Maximum Queue Length', labelpad=10, fontsize=10)
    fig.ax.set_xscale("log", base=2)
    fig.ax.set_yscale("log", base=2)

    fig.set(ylim=(2 ** (-2), 2 ** 5))
    fig.set(xlim=(2 ** 1, 2 ** 23))
    plt.xticks([2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608],
               [f"$2^{{{i * 2 - 1}}}$" for i in range(1, 13)])

    plt.grid()

    fig.savefig(FIGURE_PATH_MAX_QUEUE)
    # plt.show()

print("finished!")
