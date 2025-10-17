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
parser.add_argument("--order_alpha", default=1 / 12, type=float)
parser.add_argument("--holding_cost", default=0.01, type=float)

args = parser.parse_args()

num_sim = args.num_sim

time_base = 1 + np.arange(args.T)

# Define the number of points you want to extract
num_points = 100

# Generate logarithmically spaced indices
log_indices = np.linspace(0, args.T - 1, num=num_points)

# Round to the nearest integer and convert to int
log_indices = np.unique(np.round(log_indices).astype(int))  # Ensure unique indices

time_base = time_base[log_indices]

FIGURE_PATH_COMP = os.path.join(CUR_DIR, "results_multi",
                                "comparison_multi_link_holding_cost_" + str(args.holding_cost) + ".pdf")
FIGURE_PATH_PERCENT = os.path.join(CUR_DIR, "results_multi",
                                   "comparison_percent_multi_link_holding_cost_" + str(args.holding_cost) + ".pdf")

'''
known_two_price (Varma), learn_threshold (Yang), learn_two_price_threshold (proposed),
'''
policy_list = ["known_two_price", "learn_threshold", "learn_two_price_threshold"]
algorithm_list = ["Two-price policy - known\n(Varma et al., 2023)", "Threshold policy\n(Yang & Ying, 2024)",
                  "Probabilistic two-price policy\n(Proposed)"]
pd_data_comp_list = []
for i in range(len(policy_list)):
    policy = policy_list[i]
    PATH_PROFIT = os.path.join(CUR_DIR,
                               "data_multi/profit_"
                               + policy
                               + "_alpha_"
                               + "{:.4f}".format(args.order_alpha)
                               + "_T_" + str(args.T)
                               + ".npy")
    PATH_AVG_QUEUE = os.path.join(CUR_DIR,
                                  "data_multi/avg_queue_"
                                  + policy
                                  + "_alpha_"
                                  + "{:.4f}".format(args.order_alpha)
                                  + "_T_" + str(args.T)
                                  + ".npy")
    profit_regret = np.load(PATH_PROFIT)
    avg_queue = np.load(PATH_AVG_QUEUE)

    profit_regret = profit_regret[log_indices, 0:num_sim]
    avg_queue = avg_queue[log_indices, 0:num_sim]

    comp = profit_regret + args.holding_cost * time_base.reshape((-1, 1)) * avg_queue
    pd_data_comp = pd.DataFrame(np.concatenate((time_base.repeat(num_sim).reshape((-1, 1)),
                                                comp.reshape((-1, 1))), axis=1),
                                columns=["Time", "Objective"])

    pd_data_comp["Algorithm"] = algorithm_list[i]

    pd_data_comp_list.append(pd_data_comp)

pd_data_comp_total = pd.concat(pd_data_comp_list, axis=0, ignore_index=True)

pd_data_comp_percent = pd_data_comp_list[2]
denominator = pd_data_comp_list[1]["Objective"].abs()
denominator[denominator == 0] = 0.01
pd_data_comp_percent["Percent"] = (pd_data_comp_list[1]["Objective"] - pd_data_comp_list[2]["Objective"]) / denominator

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Objective", markers=False, data=pd_data_comp_total, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="Algorithm", style="Algorithm",
                      # legend=False,
                      height=4,
                      aspect=1.3
                      )
    if args.holding_cost == 0.01:
        sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.14, 0.92), title=None, frameon=True)  # for weight 0.01
    else:
        sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.21, 0.96), title=None, frameon=True)  # for weight 0.001
    fig.set_axis_labels('Time', 'Profit Regret + ' + str(args.holding_cost) + r'$\times$ Queue Length', labelpad=10,
                        fontsize=10)

    fig.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if args.holding_cost == 0.01:
        fig.set(ylim=(0, 3e6))  # for weight 0.01
    elif args.holding_cost == 0.005:
        fig.set(ylim=(0, 2e6))  # for weight 0.005
    else:
        fig.set(ylim=(0, 14e5))  # for weight 0.001
    fig.set(xlim=(1, args.T))

    plt.grid()

    fig.savefig(FIGURE_PATH_COMP)
    # plt.show()

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Percent", markers=False, data=pd_data_comp_percent, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="Algorithm", style="Algorithm",
                      legend=False,
                      height=4,
                      aspect=1.3
                      )
    # sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.285, 0.50), title=None, frameon=True)
    fig.set_axis_labels('Time', 'Percentage Improvement of' + '\n(Profit Regret + ' + str(
        args.holding_cost) + r'$\times$ Queue Length)', labelpad=10, fontsize=10)

    if args.holding_cost == 0.01:
        fig.set(ylim=(-1.0, 0.6))  # for weight 0.01
    elif args.holding_cost == 0.005:
        fig.set(ylim=(-1.0, 0.6))  # for weight 0.005
    else:
        fig.set(ylim=(-3.0, 0.0))  # for weight 0.001
    fig.set(xlim=(1, args.T))

    plt.grid()

    fig.savefig(FIGURE_PATH_PERCENT)
    # plt.show()

print("finished!")
