import numpy as np
import os
import inspect
import matplotlib
import matplotlib.pyplot as plt
import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

matplotlib.use('TkAgg')

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--num_sim", default=10, type=int)
parser.add_argument("--T", default=1000000, type=int)
parser.add_argument("--policy", default="learn_two_price_threshold")

args = parser.parse_args()

num_sim = args.num_sim

order_alpha = [5 / 120, 6 / 120, 7 / 120, 8 / 120, 9 / 120, 10 / 120]
regret_list = []
avg_queue_list = []
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

    profit_regret_order = np.mean(np.log2(profit_regret) / np.log2(time_indices).reshape((-1, 1)))
    avg_queue_order = np.mean(np.log2(avg_queue) / np.log2(time_indices).reshape((-1, 1)))

    regret_list.append(profit_regret_order)
    avg_queue_list.append(avg_queue_order)


order_alpha = [i * 2 for i in order_alpha]

# Fit a line (degree 1) to the data
slope, intercept = np.polyfit(order_alpha, regret_list, 1)

# Print the slope and intercept
print(f"Slope: {slope}, Intercept: {intercept}")

# Fit a line (degree 1) to the data
slope2, intercept2 = np.polyfit(order_alpha, avg_queue_list, 1)

# Print the slope and intercept
print(f"Slope: {slope2}, Intercept: {intercept2}")


# Plot the data
plt.scatter(order_alpha, regret_list, label="Data points")

# Plot the fitted line
y_fit = np.polyval([slope, intercept], order_alpha)  # Calculate fitted values
plt.plot(order_alpha, y_fit, label="Fitted line", color="red")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# Plot the data
plt.scatter(order_alpha, avg_queue_list, label="Data points")

# Plot the fitted line
y_fit = np.polyval([slope2, intercept2], order_alpha)  # Calculate fitted values
plt.plot(order_alpha, y_fit, label="Fitted line", color="red")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


print("finished!")
