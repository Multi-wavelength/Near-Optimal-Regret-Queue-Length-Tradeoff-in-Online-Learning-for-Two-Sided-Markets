import numpy as np
import os
import inspect
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

matplotlib.use('TkAgg')

plt.rcParams.update({
    "font.size": 100,       # Base font size for all text
    "axes.labelsize": 120,  # Font size for axis labels
    "axes.titlesize": 100,  # Font size for titles
    "xtick.labelsize": 100, # Font size for x-axis tick labels
    "ytick.labelsize": 100, # Font size for y-axis tick labels
    "legend.fontsize": 100, # Font size for legends
    "figure.titlesize": 100 # Font size for figure title
})

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--num_sim", default=10, type=int)
parser.add_argument("--T", default=1000000, type=int)
parser.add_argument("--policy", default="learn_two_price_threshold")
'''
known_two_price (Varma), learn_threshold (Yang), learn_two_price_threshold (proposed),
'''

args = parser.parse_args()

num_sim = args.num_sim

FIGURE_PATH_TRADEOFF = os.path.join(CUR_DIR, "results_single", "regret_queue_tradeoff.pdf")

order_alpha = [5 / 120, 6 / 120, 7 / 120, 8 / 120, 9 / 120, 10 / 120]
pd_data_regret_list = []
pd_data_avg_queue_list = []
time_indices = np.arange(args.T // 10, args.T)
for _order_alpha in order_alpha:
    PATH_PROFIT = os.path.join(CUR_DIR,
                               "data_single/profit_"
                               + args.policy
                               + "_alpha_"
                               + "{:.4f}".format(_order_alpha)
                               + "_T_" + str(args.T)
                               + ".npy")
    PATH_AVG_QUEUE = os.path.join(CUR_DIR,
                                  "data_single/avg_queue_"
                                  + args.policy
                                  + "_alpha_"
                                  + "{:.4f}".format(_order_alpha)
                                  + "_T_" + str(args.T)
                                  + ".npy")

    profit_regret = np.load(PATH_PROFIT)
    avg_queue = np.load(PATH_AVG_QUEUE)

    profit_regret = profit_regret[time_indices, 0:num_sim]
    avg_queue = avg_queue[time_indices, 0:num_sim]

    profit_regret_order = np.mean(np.log2(profit_regret) / np.log2(time_indices).reshape((-1, 1)), axis=0)
    avg_queue_order = np.mean(np.log2(avg_queue) / np.log2(time_indices).reshape((-1, 1)), axis=0)

    pd_data_regret = pd.DataFrame(profit_regret_order, columns=["Profit Regret"])
    pd_data_avg_queue = pd.DataFrame(avg_queue_order, columns=["Average Queue Length"])

    pd_data_regret["gamma"] = _order_alpha * 2
    pd_data_avg_queue["gamma"] = _order_alpha * 2

    pd_data_regret_list.append(pd_data_regret)
    pd_data_avg_queue_list.append(pd_data_avg_queue)

pd_data_regret_total = pd.concat(pd_data_regret_list, axis=0, ignore_index=True)
pd_data_avg_queue_total = pd.concat(pd_data_avg_queue_list, axis=0, ignore_index=True)

# Create a figure and primary axis
fig, ax1 = plt.subplots(figsize=(55, 40), dpi=400)

# Define custom colors and line styles
color1 = "tab:blue"
color2 = "tab:red"
linestyle1 = "-"
linestyle2 = "--"

with sns.plotting_context("notebook"):
    # Plot the first dataset (Profit Regret) on ax1 (Left Y-axis)
    sns.lineplot(x="gamma", y="Profit Regret",
                 # errorbar=None,
                 data=pd_data_regret_total,
                 # marker="o",
                 markers=True,
                 label="Profit Regret",
                 linestyle=linestyle1, color=color1, ax=ax1)

# Configure ax1
ax1.set_xlabel(r"$\gamma$")
ax1.set_ylabel(r"$\frac{\log R(t)}{\log t}$", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(True)
# ax1.set_ylabel("Profit Regret", fontsize=10)
ax1.set_ylim(0.66, 0.82)
ax1.set_xlim(0.08, 0.17)
# Set custom x-axis ticks
xticks = [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]  # Define your custom tick positions
ax1.set_xticks(xticks)  # Apply the ticks to the x-axis

# Create secondary y-axis
ax2 = ax1.twinx()

with sns.plotting_context("notebook"):
    # Plot the second dataset (Average Queue Length) on ax2 (Right Y-axis)
    sns.lineplot(x="gamma", y="Average Queue Length",
                 # errorbar=None,
                 data=pd_data_avg_queue_total,
                 markers=True,
                 # marker="s",
                 label="Average Queue Length",
                 legend=False,
                 linestyle=linestyle2, color=color2, ax=ax2)

# Configure ax2
# ax2.set_ylabel("Average Queue Length", fontsize=10)
ax2.set_ylim(0.02, 0.18)
ax2.set_ylabel(r"$\frac{\log \text{AvgQLen}(t)}{\log t}$", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.25, 0.99))

# Save the combined figure
fig.savefig(FIGURE_PATH_TRADEOFF, bbox_inches="tight")

# Show the plot
# plt.show()

print("finished!")
