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
parser.add_argument("--T", default=1000000, type=int)
parser.add_argument("--denominator_alpha", default=12, type=int)
parser.add_argument("--num_sim", default=10, type=int)
parser.add_argument("--a_min", default=0.01, type=float)
parser.add_argument("--beta", default=1.0, type=float)  # originally beta = 1.0
parser.add_argument("--epsilon_scaling", default=1.0, type=float)  # originally epsilon_scaling = 1.0
parser.add_argument("--delta_scaling", default=0.2, type=float)  # originally delta_scaling = 0.2
parser.add_argument("--alpha_scaling", default=0.2, type=float)  # originally alpha_scaling = 0.2
parser.add_argument("--eta_scaling", default=0.2, type=float)  # originally eta_scaling = 0.2
parser.add_argument("--ecs_scaling", default=6.0, type=float)  # originally ecs_scaling = 6.0
parser.add_argument("--prob", default=0.5, type=float)  # originally = 0.5
parser.add_argument("--policy", default="learn_two_price_threshold")
parser.add_argument("--parameter", default="epsilon_scaling")  # "epsilon_scaling", "delta_scaling", "eta_scaling"


args = parser.parse_args()

num_sim = args.num_sim

time_base = 1 + np.arange(args.T)

# Define the number of points you want to extract
num_points = 100

# Generate logarithmically spaced indices
log_indices = np.logspace(0, np.log10(args.T - 1), num=num_points, base=10)

# Round to the nearest integer and convert to int
log_indices = np.unique(np.round(log_indices).astype(int))  # Ensure unique indices

time_base = time_base[log_indices]

FIGURE_PATH_REGRET = os.path.join(CUR_DIR, "results_single", "profit_regret_" + args.parameter + ".pdf")
FIGURE_PATH_AVG_QUEUE = os.path.join(CUR_DIR, "results_single", "avg_queue_" + args.parameter + ".pdf")
FIGURE_PATH_MAX_QUEUE = os.path.join(CUR_DIR, "results_single", "max_queue_" + args.parameter + ".pdf")


beta = args.beta
T = args.T
epsilon_scaling = args.epsilon_scaling
delta_scaling = args.delta_scaling
alpha_scaling = args.alpha_scaling
eta_scaling = args.eta_scaling
probability = args.prob
ecs_scaling = args.ecs_scaling


'''
known_two_price (Varma), learn_threshold (Yang), learn_two_price_threshold (proposed),
'''
policy = "learn_two_price_threshold"
algorithm = "Probabilistic two-price policy\n(Proposed)"
pd_data_regret_list = []
pd_data_avg_queue_list = []
pd_data_max_queue_list = []

if args.parameter == "epsilon_scaling":
    parameter_list = [0.6, 0.8, 1.0, 1.2, 1.4]
    legend = r"$\epsilon$ coefficient"
elif args.parameter == "delta_scaling":
    parameter_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    legend = r"$\delta$ coefficient"
elif args.parameter == "eta_scaling":
    parameter_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    legend = r"$\eta$ coefficient"


for para in parameter_list:
    if args.parameter == "epsilon_scaling":
        filename_string = ("_beta_" + str(beta)
                           + "_epsilon_scaling_" + str(para)
                           + "_delta_scaling_" + str(delta_scaling)
                           + "_alpha_scaling_" + str(alpha_scaling)
                           + "_eta_scaling_" + str(eta_scaling)
                           + "_prob_" + str(probability))
    elif args.parameter == "delta_scaling":
        filename_string = ("_beta_" + str(beta)
                           + "_epsilon_scaling_" + str(epsilon_scaling)
                           + "_delta_scaling_" + str(para)
                           + "_alpha_scaling_" + str(alpha_scaling)
                           + "_eta_scaling_" + str(eta_scaling)
                           + "_prob_" + str(probability))
    elif args.parameter == "eta_scaling":
        filename_string = ("_beta_" + str(beta)
                           + "_epsilon_scaling_" + str(epsilon_scaling)
                           + "_delta_scaling_" + str(delta_scaling)
                           + "_alpha_scaling_" + str(alpha_scaling)
                           + "_eta_scaling_" + str(para)
                           + "_prob_" + str(probability))
    PATH_PROFIT = os.path.join(CUR_DIR,
                               "data_single/profit_"
                               + args.policy
                               + "_alpha_1_over_"
                               + str(args.denominator_alpha)
                               + filename_string
                               + ".npy")
    PATH_AVG_QUEUE = os.path.join(CUR_DIR,
                                  "data_single/avg_queue_"
                                  + args.policy
                                  + "_alpha_1_over_"
                                  + str(args.denominator_alpha)
                                  + filename_string
                                  + ".npy")
    PATH_MAX_QUEUE = os.path.join(CUR_DIR,
                                  "data_single/max_queue_"
                                  + args.policy
                                  + "_alpha_1_over_"
                                  + str(args.denominator_alpha)
                                  + filename_string
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

    pd_data_regret["parameter scaling"] = str(para)
    pd_data_avg_queue["parameter scaling"] = str(para)
    pd_data_max_queue["parameter scaling"] = str(para)

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
                      hue="parameter scaling", style="parameter scaling",
                      height=4,
                      aspect=1.3
                      # legend=False
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.21, 0.95), title=legend, frameon=True)
    fig.set_axis_labels('Time', 'Profit Regret', labelpad=10, fontsize=10)
    fig.ax.set_xscale("log", base=2)
    fig.ax.set_yscale("log", base=2)

    fig.set(ylim=(2 ** 2, 2 ** 16))
    fig.set(xlim=(2 ** 7, 2 ** 19))

    plt.grid()

    fig.savefig(FIGURE_PATH_REGRET)
    # plt.show()

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Average Queue Length", markers=False, data=pd_data_avg_queue_total, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="parameter scaling", style="parameter scaling",
                      height=4,
                      aspect=1.3
                      # legend=False
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.14, 0.96), title=legend, frameon=True)
    fig.set_axis_labels('Time', 'Average Queue Length', labelpad=10, fontsize=10)
    fig.ax.set_xscale("log", base=2)
    fig.ax.set_yscale("log", base=2)

    fig.set(ylim=(2 ** (-1), 2 ** 3))
    fig.set(xlim=(2 ** 7, 2 ** 19))

    plt.grid()

    fig.savefig(FIGURE_PATH_AVG_QUEUE)
    # plt.show()

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Maximum Queue Length", markers=False, data=pd_data_max_queue_total, kind='line',
                      # errorbar=None,
                      # errorbar="sd",
                      hue="parameter scaling", style="parameter scaling",
                      height=4,
                      aspect=1.3
                      # legend=False
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.13, 0.96), title=legend, frameon=True)
    fig.set_axis_labels('Time', 'Maximum Queue Length', labelpad=10, fontsize=10)
    fig.ax.set_xscale("log", base=2)
    fig.ax.set_yscale("log", base=2)

    fig.set(ylim=(2 ** 0, 2 ** 5))
    fig.set(xlim=(2 ** 7, 2 ** 19))

    plt.grid()

    fig.savefig(FIGURE_PATH_MAX_QUEUE)
    # plt.show()

print("finished!")
