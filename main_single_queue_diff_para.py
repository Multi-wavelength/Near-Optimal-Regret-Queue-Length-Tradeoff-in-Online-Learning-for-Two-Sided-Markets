import numpy as np
import cvxpy as cp
import inspect
from fractions import Fraction
import os
import argparse
from joblib import Parallel, delayed

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
'''
known_two_price (Varma), learn_threshold (Yang), learn_two_price_threshold (proposed),
'''

args = parser.parse_args()

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

threshold_frac = Fraction(2, args.denominator_alpha)
threshold = float(threshold_frac)
threshold2_frac = Fraction(1, args.denominator_alpha)
threshold2 = float(threshold2_frac)
beta = args.beta
T = args.T
epsilon_scaling = args.epsilon_scaling
delta_scaling = args.delta_scaling
alpha_scaling = args.alpha_scaling
eta_scaling = args.eta_scaling
probability = args.prob
ecs_scaling = args.ecs_scaling
PATH_PROFIT = os.path.join(CUR_DIR,
                           "data_single/profit_"
                           + args.policy
                           + "_alpha_1_over_"
                           + str(args.denominator_alpha)
                           + "_beta_" + str(beta)
                           + "_epsilon_scaling_" + str(epsilon_scaling)
                           + "_delta_scaling_" + str(delta_scaling)
                           + "_alpha_scaling_" + str(alpha_scaling)
                           + "_eta_scaling_" + str(eta_scaling)
                           + "_prob_" + str(probability)
                           + ".npy")
PATH_AVG_QUEUE = os.path.join(CUR_DIR,
                              "data_single/avg_queue_"
                              + args.policy
                              + "_alpha_1_over_"
                              + str(args.denominator_alpha)
                              + "_beta_" + str(beta)
                              + "_epsilon_scaling_" + str(epsilon_scaling)
                              + "_delta_scaling_" + str(delta_scaling)
                              + "_alpha_scaling_" + str(alpha_scaling)
                              + "_eta_scaling_" + str(eta_scaling)
                              + "_prob_" + str(probability)
                              + ".npy")
PATH_MAX_QUEUE = os.path.join(CUR_DIR,
                              "data_single/max_queue_"
                              + args.policy
                              + "_alpha_1_over_"
                              + str(args.denominator_alpha)
                              + "_beta_" + str(beta)
                              + "_epsilon_scaling_" + str(epsilon_scaling)
                              + "_delta_scaling_" + str(delta_scaling)
                              + "_alpha_scaling_" + str(alpha_scaling)
                              + "_eta_scaling_" + str(eta_scaling)
                              + "_prob_" + str(probability)
                              + ".npy")


# Define demand and supply functions
def demand_curve(p):
    return 2 * (2 - p) / 4  # 1 - p / 2


def supply_curve(p):
    return 2 * p / 4  # p / 2


def inv_demand_curve(arr_rate):
    return 2 * (1 - arr_rate)


def inv_supply_curve(arr_rate):
    return 2 * arr_rate


# Fluid solution
def solve_fluid_model(E, num_customer, num_server, silence=True):
    """
    With demand function and supply functions, lambda = 1 - p / 2, mu = p / 2, and the compatibility matrix,
    find the optimal solution of the fluid problem
    """
    E = [(i, j) for i in range(num_customer) for j in range(num_server) if E[i, j] == 1]
    x = {(i, j): cp.Variable(nonneg=True) for i, j in E}
    arr_rate_demand = [cp.sum([x[i, j] for j in range(num_server) if (i, j) in E]) for i in range(num_customer)]
    arr_rate_supply = [cp.sum([x[i, j] for i in range(num_customer) if (i, j) in E]) for j in range(num_server)]

    obj = ((cp.sum([2 * demand - 2 * cp.square(demand) for demand in arr_rate_demand]))
           - cp.sum([2 * cp.square(supply) for supply in arr_rate_supply]))

    # Define the constraints
    constraints = []
    for i in range(num_customer):
        constraints.append(cp.sum([x[i, j] for j in range(num_server) if (i, j) in E]) <= 1)
        constraints.append(cp.sum([x[i, j] for j in range(num_server) if (i, j) in E]) >= 0)

    for j in range(num_server):
        constraints.append(cp.sum([x[i, j] for i in range(num_customer) if (i, j) in E]) <= 1)
        constraints.append(cp.sum([x[i, j] for i in range(num_customer) if (i, j) in E]) >= 0)

    # Define the problem
    prob = cp.Problem(cp.Maximize(obj), constraints)

    # Solve the problem
    result = prob.solve()

    x_array = np.zeros((num_customer, num_server))
    arr_c_array = np.zeros((num_customer,))
    arr_s_array = np.zeros((num_server,))
    if not silence:
        # Print the results
        print("Optimal value:", prob.value)
        print("Optimal solution:")
    for (i, j), var in x.items():
        x_array[i, j] = var.value
        if not silence:
            print(f"x[{i},{j}] = {var.value:.4f}")

    for i in range(num_customer):
        arr_c_array[i] = arr_rate_demand[i].value
    for j in range(num_server):
        arr_s_array[j] = arr_rate_supply[j].value
    return x_array, prob.value, arr_c_array, arr_s_array


def alg_known(E, num_customer, num_server, opt_value, arr_c_array, arr_s_array,
              min_p_c, max_p_c, min_p_s, max_p_s, num_sim, policy):
    """
    Assuming the demand and supply function are known,
    use the fluid pricing algorithm in the paper:
    Varma, etc., "Dynamic Pricing and Matching for Two-Sided Queues"
    :param E: the graph
    :param num_customer: number of customer types
    :param num_server: number of server types
    :param opt_value: optimal profit
    :param arr_c_array: target arrival rate for the customer side
    :param arr_s_array: target arrival rate for the server side
    :param min_p_c: minimum price for the customer side
    :param max_p_c: maximum price for the customer side
    :param min_p_s: minimum price for the server side
    :param max_p_s: maximum price for the server side
    :param num_sim: number of simulation
    :param policy: policy


    :return profit_regret: profit regret
    :return cum_queue: total queue length cumulated over time
    :return max_queue: maximum queue length
    """

    profit_regret = np.zeros((T, num_sim))
    cum_queue = np.zeros((T, num_sim))
    max_queue = np.zeros((T, num_sim))
    price_c_array = np.tile(inv_demand_curve(arr_c_array).reshape(-1, 1), (1, num_sim))
    price_s_array = np.tile(inv_supply_curve(arr_s_array).reshape(-1, 1), (1, num_sim))

    queue_c = np.zeros((num_customer, num_sim))
    queue_s = np.zeros((num_server, num_sim))
    price_c = np.zeros((num_customer, num_sim))
    price_s = np.zeros((num_server, num_sim))
    for t in range(T):
        threshold_true = (t + 1) ** threshold
        alpha = alpha_scaling * (t + 1) ** (-threshold2)
        price_c_array_small = np.tile(inv_demand_curve(np.maximum(arr_c_array - alpha, 0)).reshape(-1, 1), (1, num_sim))
        price_s_array_small = np.tile(inv_supply_curve(np.maximum(arr_s_array - alpha, 0)).reshape(-1, 1), (1, num_sim))
        if policy == "known_threshold":
            price_c[queue_c >= threshold_true] = max_p_c
            price_c[queue_c < threshold_true] = price_c_array[queue_c < threshold_true]

            price_s[queue_s >= threshold_true] = min_p_s
            price_s[queue_s < threshold_true] = price_s_array[queue_s < threshold_true]
        elif policy == "known_two_price":
            price_c[queue_c > 0] = price_c_array_small[queue_c > 0]
            price_c[queue_c == 0] = price_c_array[queue_c == 0]

            price_s[queue_s > 0] = price_s_array_small[queue_s > 0]
            price_s[queue_s == 0] = price_s_array[queue_s == 0]

        arr_prob_c = demand_curve(price_c)
        arr_prob_s = supply_curve(price_s)

        # generating random arrivals
        arrivals_c = (np.random.rand(arr_prob_c.shape[0], arr_prob_c.shape[1]) < arr_prob_c)
        arrivals_s = (np.random.rand(arr_prob_s.shape[0], arr_prob_s.shape[1]) < arr_prob_s)

        # profit
        # profit = np.sum(arr_prob_c * price_c, axis=0) - np.sum(arr_prob_s * price_s, axis=0)
        # print(mean_profit)
        profit = np.sum(arrivals_c * price_c, axis=0) - np.sum(arrivals_s * price_s, axis=0)

        # longest-queue-first matching and departure
        alg_matching_lqf(E, queue_c, queue_s, arrivals_c, arrivals_s)

        # Update metrics
        if t != T - 1:
            cum_queue[t + 1, :] = cum_queue[t, :] + np.sum(queue_c, axis=0) + np.sum(queue_s, axis=0)
            max_queue[t + 1, :] = np.maximum(max_queue[t, :],
                                             np.maximum(np.max(queue_c, axis=0), np.max(queue_s, axis=0)))
        if t > 0:
            profit_regret[t, :] = profit_regret[t - 1, :] + opt_value - profit
        else:
            profit_regret[0, :] = opt_value - profit

    return profit_regret, cum_queue, max_queue


def alg_learning_single_simu(num_customer, num_server, E,
                             policy,
                             min_p_c, max_p_c, min_p_s, max_p_s,
                             opt_value,
                             a_min, a_min_p1_div_2Nij, a_min_p1_div_2Nij_sumi, a_min_p1_div_2Nij_sumj, r):
    rng = np.random.default_rng()
    profit_regret = np.zeros((T,))
    cum_queue = np.zeros((T,))
    max_queue = np.zeros((T,))
    queue_c = np.zeros((num_customer,))
    queue_s = np.zeros((num_server,))
    price_c = np.zeros((num_customer,))
    price_s = np.zeros((num_server,))

    ecs_0 = ecs_scaling * delta_scaling

    x = delta_scaling * np.ones((num_customer, num_server)) * E

    price_c_l = np.zeros((num_customer,))
    price_c_u = np.zeros((num_customer,))
    price_s_l = np.zeros((num_server,))
    price_s_u = np.zeros((num_server,))
    price_c_l[0] = inv_demand_curve(delta_scaling) - ecs_0
    price_c_u[0] = inv_demand_curve(delta_scaling) + ecs_0
    price_s_l[0] = inv_supply_curve(delta_scaling) - ecs_0
    price_s_u[0] = inv_supply_curve(delta_scaling) + ecs_0

    t = 0
    k = 0
    threshold_true = 1
    alpha = 2 * alpha_scaling
    eta = eta_scaling
    delta = delta_scaling
    epsilon = epsilon_scaling
    ecs = ecs_scaling * max(delta, epsilon, eta)
    M_temp = int(np.ceil(np.log2(0.5 * min(2 * ecs, 2) / epsilon)))
    if M_temp < 1:
        M = 1
    else:
        M = M_temp
    N = int(np.ceil(beta / (epsilon ** 2)))
    if N < 1:
        N = 1

    price_c_l_plus = np.copy(price_c_l)
    price_c_l_minus = np.copy(price_c_l)

    price_c_u_plus = np.copy(price_c_u)
    price_c_u_minus = np.copy(price_c_u)

    price_s_l_plus = np.copy(price_s_l)
    price_s_l_minus = np.copy(price_s_l)

    price_s_u_plus = np.copy(price_s_u)
    price_s_u_minus = np.copy(price_s_u)

    price_c_l_plus[price_c_l_plus < min_p_c] = min_p_c
    price_c_u_plus[price_c_u_plus > max_p_c] = max_p_c
    price_s_l_plus[price_s_l_plus < min_p_s] = min_p_s
    price_s_u_plus[price_s_u_plus > max_p_s] = max_p_s

    price_c_l_minus[price_c_l_minus < min_p_c] = min_p_c
    price_c_u_minus[price_c_u_minus > max_p_c] = max_p_c
    price_s_l_minus[price_s_l_minus < min_p_s] = min_p_s
    price_s_u_minus[price_s_u_minus > max_p_s] = max_p_s

    while t < T:
        # print("Percentage completed: ", t / T * 100)
        u = rng.standard_normal(num_customer * num_server) * E.flatten()
        u /= np.linalg.norm(u)
        u = u.reshape((num_customer, num_server))
        x_plus = x + delta * u
        x_minus = x - delta * u

        arr_c_array_plus = np.sum(x_plus, axis=1)
        arr_c_array_minus = np.sum(x_minus, axis=1)

        arr_s_array_plus = np.sum(x_plus, axis=0)
        arr_s_array_minus = np.sum(x_minus, axis=0)

        # Bisection for "plus"
        for m in range(M):
            price_c_mid_plus = (price_c_l_plus + price_c_u_plus) / 2
            price_s_mid_plus = (price_s_l_plus + price_s_u_plus) / 2

            num_samples_c = np.zeros((num_customer,))
            num_samples_s = np.zeros((num_server,))
            sum_arr_c = np.zeros((num_customer,))
            sum_arr_s = np.zeros((num_server,))

            while np.minimum(np.min(num_samples_c), np.min(num_samples_s)) < N:
                if policy == "learn_threshold":
                    price_c[queue_c >= threshold_true] = max_p_c
                    price_c[queue_c < threshold_true] = price_c_mid_plus[queue_c < threshold_true]

                    price_s[queue_s >= threshold_true] = min_p_s
                    price_s[queue_s < threshold_true] = price_s_mid_plus[queue_s < threshold_true]
                elif policy == "learn_two_price_threshold":
                    rnd_c = (rng.random(num_customer) < probability)
                    rnd_s = (rng.random(num_server) < probability)

                    price_c[queue_c >= threshold_true] = max_p_c
                    price_c[rnd_c & (queue_c < threshold_true) & (queue_c > 0)] = price_c_mid_plus[
                        rnd_c & (queue_c < threshold_true) & (queue_c > 0)]
                    price_c[(~rnd_c) & (queue_c < threshold_true) & (queue_c > 0)] = np.minimum(
                        price_c_mid_plus[(~rnd_c) & (queue_c < threshold_true) & (queue_c > 0)] + alpha
                        , max_p_c)
                    price_c[queue_c == 0] = price_c_mid_plus[queue_c == 0]

                    price_s[queue_s >= threshold_true] = min_p_s
                    price_s[rnd_s & (queue_s < threshold_true) & (queue_s > 0)] = price_s_mid_plus[
                        rnd_s & (queue_s < threshold_true) & (queue_s > 0)]
                    price_s[(~rnd_s) & (queue_s < threshold_true) & (queue_s > 0)] = np.maximum(
                        price_s_mid_plus[(~rnd_s) & (queue_s < threshold_true) & (queue_s > 0)] - alpha
                        , min_p_s)

                    price_s[queue_s == 0] = price_s_mid_plus[queue_s == 0]

                arr_prob_c = demand_curve(price_c)
                arr_prob_s = supply_curve(price_s)

                # generating random arrivals
                arrivals_c = (rng.random(arr_prob_c.shape[0]) < arr_prob_c)
                arrivals_s = (rng.random(arr_prob_s.shape[0]) < arr_prob_s)

                # collecting samples
                if policy == "learn_threshold":
                    num_samples_c[queue_c < threshold_true] = num_samples_c[queue_c < threshold_true] + 1
                    sum_arr_c[queue_c < threshold_true] = sum_arr_c[queue_c < threshold_true] + arrivals_c[
                        queue_c < threshold_true]
                    num_samples_s[queue_s < threshold_true] = num_samples_s[queue_s < threshold_true] + 1
                    sum_arr_s[queue_s < threshold_true] = sum_arr_s[queue_s < threshold_true] + arrivals_s[
                        queue_s < threshold_true]
                elif policy == "learn_two_price_threshold":
                    num_samples_c[rnd_c & (queue_c < threshold_true)] = num_samples_c[
                                                                            rnd_c & (queue_c < threshold_true)] + 1
                    sum_arr_c[rnd_c & (queue_c < threshold_true)] = sum_arr_c[rnd_c & (queue_c < threshold_true)] + \
                                                                    arrivals_c[
                                                                        rnd_c & (queue_c < threshold_true)]
                    num_samples_s[rnd_s & (queue_s < threshold_true)] = num_samples_s[
                                                                            rnd_s & (queue_s < threshold_true)] + 1
                    sum_arr_s[rnd_s & (queue_s < threshold_true)] = sum_arr_s[rnd_s & (queue_s < threshold_true)] + \
                                                                    arrivals_s[
                                                                        rnd_s & (queue_s < threshold_true)]

                # profit
                profit = np.sum(arrivals_c * price_c) - np.sum(arrivals_s * price_s)

                # longest-queue-first matching and departure
                alg_matching_lqf(E, queue_c[:, np.newaxis], queue_s[:, np.newaxis], arrivals_c[:, np.newaxis],
                                 arrivals_s[:, np.newaxis])

                # Update metrics
                if t != T - 1:
                    cum_queue[t + 1] = cum_queue[t] + np.sum(queue_c) + np.sum(queue_s)
                    max_queue[t + 1] = np.maximum(max_queue[t],
                                                  np.maximum(np.max(queue_c), np.max(queue_s)))
                if t > 0:
                    profit_regret[t] = profit_regret[t - 1] + opt_value - profit
                else:
                    profit_regret[0] = opt_value - profit

                t = t + 1
                threshold_true = (t + 1) ** threshold
                alpha = 2 * alpha_scaling * (t + 1) ** (-threshold2)

                if t >= T:
                    break

            if t >= T:
                break

            avg_arr_rate_c = sum_arr_c / num_samples_c
            avg_arr_rate_s = sum_arr_s / num_samples_s

            price_c_l_plus[avg_arr_rate_c > arr_c_array_plus] = price_c_mid_plus[avg_arr_rate_c > arr_c_array_plus]
            price_c_u_plus[avg_arr_rate_c <= arr_c_array_plus] = price_c_mid_plus[avg_arr_rate_c <= arr_c_array_plus]

            price_s_u_plus[avg_arr_rate_s > arr_s_array_plus] = price_s_mid_plus[avg_arr_rate_s > arr_s_array_plus]
            price_s_l_plus[avg_arr_rate_s <= arr_s_array_plus] = price_s_mid_plus[avg_arr_rate_s <= arr_s_array_plus]

        if t >= T:
            break

        # Bisection for "minus"
        for m in range(M):
            price_c_mid_minus = (price_c_l_minus + price_c_u_minus) / 2
            price_s_mid_minus = (price_s_l_minus + price_s_u_minus) / 2

            num_samples_c = np.zeros((num_customer,))
            num_samples_s = np.zeros((num_server,))
            sum_arr_c = np.zeros((num_customer,))
            sum_arr_s = np.zeros((num_server,))

            while np.minimum(np.min(num_samples_c), np.min(num_samples_s)) < N:
                if policy == "learn_threshold":
                    price_c[queue_c >= threshold_true] = max_p_c
                    price_c[queue_c < threshold_true] = price_c_mid_minus[queue_c < threshold_true]

                    price_s[queue_s >= threshold_true] = min_p_s
                    price_s[queue_s < threshold_true] = price_s_mid_minus[queue_s < threshold_true]
                elif policy == "learn_two_price_threshold":
                    rnd_c = (rng.random(num_customer) < probability)
                    rnd_s = (rng.random(num_server) < probability)

                    price_c[queue_c >= threshold_true] = max_p_c
                    price_c[rnd_c & (queue_c < threshold_true) & (queue_c > 0)] = price_c_mid_minus[
                        rnd_c & (queue_c < threshold_true) & (queue_c > 0)]
                    price_c[(~rnd_c) & (queue_c < threshold_true) & (queue_c > 0)] = np.minimum(
                        price_c_mid_minus[(~rnd_c) & (queue_c < threshold_true) & (queue_c > 0)] + alpha
                        , max_p_c)
                    price_c[queue_c == 0] = price_c_mid_minus[queue_c == 0]

                    price_s[queue_s >= threshold_true] = min_p_s
                    price_s[rnd_s & (queue_s < threshold_true) & (queue_s > 0)] = price_s_mid_minus[
                        rnd_s & (queue_s < threshold_true) & (queue_s > 0)]
                    price_s[(~rnd_s) & (queue_s < threshold_true) & (queue_s > 0)] = np.maximum(
                        price_s_mid_minus[(~rnd_s) & (queue_s < threshold_true) & (queue_s > 0)] - alpha
                        , min_p_s)

                    price_s[queue_s == 0] = price_s_mid_minus[queue_s == 0]

                arr_prob_c = demand_curve(price_c)
                arr_prob_s = supply_curve(price_s)

                # generating random arrivals
                arrivals_c = (rng.random(arr_prob_c.shape[0]) < arr_prob_c)
                arrivals_s = (rng.random(arr_prob_s.shape[0]) < arr_prob_s)

                # collecting samples
                if policy == "learn_threshold":
                    num_samples_c[queue_c < threshold_true] = num_samples_c[queue_c < threshold_true] + 1
                    sum_arr_c[queue_c < threshold_true] = sum_arr_c[queue_c < threshold_true] + arrivals_c[
                        queue_c < threshold_true]
                    num_samples_s[queue_s < threshold_true] = num_samples_s[queue_s < threshold_true] + 1
                    sum_arr_s[queue_s < threshold_true] = sum_arr_s[queue_s < threshold_true] + arrivals_s[
                        queue_s < threshold_true]
                elif policy == "learn_two_price_threshold":
                    num_samples_c[rnd_c & (queue_c < threshold_true)] = num_samples_c[
                                                                            rnd_c & (queue_c < threshold_true)] + 1
                    sum_arr_c[rnd_c & (queue_c < threshold_true)] = sum_arr_c[rnd_c & (queue_c < threshold_true)] + \
                                                                    arrivals_c[
                                                                        rnd_c & (queue_c < threshold_true)]
                    num_samples_s[rnd_s & (queue_s < threshold_true)] = num_samples_s[
                                                                            rnd_s & (queue_s < threshold_true)] + 1
                    sum_arr_s[rnd_s & (queue_s < threshold_true)] = sum_arr_s[rnd_s & (queue_s < threshold_true)] + \
                                                                    arrivals_s[
                                                                        rnd_s & (queue_s < threshold_true)]

                # profit
                profit = np.sum(arrivals_c * price_c) - np.sum(arrivals_s * price_s)

                # longest-queue-first matching and departure
                alg_matching_lqf(E, queue_c[:, np.newaxis], queue_s[:, np.newaxis], arrivals_c[:, np.newaxis],
                                 arrivals_s[:, np.newaxis])

                # Update metrics
                if t != T - 1:
                    cum_queue[t + 1] = cum_queue[t] + np.sum(queue_c) + np.sum(queue_s)
                    max_queue[t + 1] = np.maximum(max_queue[t],
                                                  np.maximum(np.max(queue_c), np.max(queue_s)))
                if t > 0:
                    profit_regret[t] = profit_regret[t - 1] + opt_value - profit
                else:
                    profit_regret[0] = opt_value - profit

                t = t + 1
                threshold_true = (t + 1) ** threshold
                alpha = 2 * alpha_scaling * (t + 1) ** (-threshold2)

                if t >= T:
                    break
            if t >= T:
                break

            avg_arr_rate_c = sum_arr_c / num_samples_c
            avg_arr_rate_s = sum_arr_s / num_samples_s

            price_c_l_minus[avg_arr_rate_c > arr_c_array_minus] = price_c_mid_minus[avg_arr_rate_c > arr_c_array_minus]
            price_c_u_minus[avg_arr_rate_c <= arr_c_array_minus] = price_c_mid_minus[
                avg_arr_rate_c <= arr_c_array_minus]

            price_s_u_minus[avg_arr_rate_s > arr_s_array_minus] = price_s_mid_minus[avg_arr_rate_s > arr_s_array_minus]
            price_s_l_minus[avg_arr_rate_s <= arr_s_array_minus] = price_s_mid_minus[
                avg_arr_rate_s <= arr_s_array_minus]

        if t >= T:
            break

        if M_temp > 0:
            price_c_mid_plus = (price_c_l_plus + price_c_u_plus) / 2
            price_s_mid_plus = (price_s_l_plus + price_s_u_plus) / 2
            price_c_mid_minus = (price_c_l_minus + price_c_u_minus) / 2
            price_s_mid_minus = (price_s_l_minus + price_s_u_minus) / 2

            # estimation of the gradient
            profit_plus = (np.sum(arr_c_array_plus * price_c_mid_plus)
                           - np.sum(arr_s_array_plus * price_s_mid_plus))
            profit_minus = (np.sum(arr_c_array_minus * price_c_mid_minus)
                            - np.sum(arr_s_array_minus * price_s_mid_minus))

            # projected gradient ascent
            temp = x + eta * (profit_plus - profit_minus) * np.sum(E) / delta / 2 * u

            if E.shape[0] == 1 and E.shape[1] == 1:
                r = (1 - a_min) / 2
                assert delta < r, "delta should be smaller than r"
                lb = (1 + a_min) / 2 - (1 - delta / r) * (1 - a_min) / 2
                ub = (1 + a_min) / 2 + (1 - delta / r) * (1 - a_min) / 2
                if temp[0, 0] < lb:
                    x[0, 0] = lb
                elif temp[0, 0] > ub:
                    x[0, 0] = ub
                else:
                    x = temp
            else:
                x = projection(E, num_customer, num_server, delta, temp, a_min, a_min_p1_div_2Nij,
                               a_min_p1_div_2Nij_sumi, a_min_p1_div_2Nij_sumj, r)

            # Update the price interval
            price_c_l_plus = price_c_mid_plus - ecs
            price_c_u_plus = price_c_mid_plus + ecs
            price_s_l_plus = price_s_mid_plus - ecs
            price_s_u_plus = price_s_mid_plus + ecs

            price_c_l_plus[price_c_l_plus < min_p_c] = min_p_c
            price_c_u_plus[price_c_u_plus > max_p_c] = max_p_c
            price_s_l_plus[price_s_l_plus < min_p_s] = min_p_s
            price_s_u_plus[price_s_u_plus > max_p_s] = max_p_s

            # Update the price interval
            price_c_l_minus = price_c_mid_minus - ecs
            price_c_u_minus = price_c_mid_minus + ecs
            price_s_l_minus = price_s_mid_minus - ecs
            price_s_u_minus = price_s_mid_minus + ecs

            price_c_l_minus[price_c_l_minus < min_p_c] = min_p_c
            price_c_u_minus[price_c_u_minus > max_p_c] = max_p_c
            price_s_l_minus[price_s_l_minus < min_p_s] = min_p_s
            price_s_u_minus[price_s_u_minus > max_p_s] = max_p_s
        else:
            price_c_l_plus = np.copy(price_c_l)
            price_c_l_minus = np.copy(price_c_l)

            price_c_u_plus = np.copy(price_c_u)
            price_c_u_minus = np.copy(price_c_u)

            price_s_l_plus = np.copy(price_s_l)
            price_s_l_minus = np.copy(price_s_l)

            price_s_u_plus = np.copy(price_s_u)
            price_s_u_minus = np.copy(price_s_u)

            price_c_l_plus[price_c_l_plus < min_p_c] = min_p_c
            price_c_u_plus[price_c_u_plus > max_p_c] = max_p_c
            price_s_l_plus[price_s_l_plus < min_p_s] = min_p_s
            price_s_u_plus[price_s_u_plus > max_p_s] = max_p_s

            price_c_l_minus[price_c_l_minus < min_p_c] = min_p_c
            price_c_u_minus[price_c_u_minus > max_p_c] = max_p_c
            price_s_l_minus[price_s_l_minus < min_p_s] = min_p_s
            price_s_u_minus[price_s_u_minus > max_p_s] = max_p_s

        # Update parameters
        eta = eta_scaling * (t + 1) ** (-threshold)
        delta = delta_scaling * (t + 1) ** (-threshold)
        epsilon = epsilon_scaling * (t + 1) ** (- 2 * threshold)
        ecs = ecs_scaling * max(delta, epsilon, eta)
        M_temp = int(np.ceil(np.log2(0.5 * min(2 * ecs, 2) / epsilon)))
        if M_temp < 1:
            M = 1
        else:
            M = M_temp
        N = int(np.ceil(beta / (epsilon ** 2)))
        if N < 1:
            N = 1

        k = k + 1
    k_array = k * np.ones((1,))

    return profit_regret, cum_queue, max_queue, k_array


def alg_learning(E, num_customer, num_server, opt_value,
                 min_p_c, max_p_c, min_p_s, max_p_s,
                 num_sim,
                 a_min, a_min_p1_div_2Nij,
                 a_min_p1_div_2Nij_sumi, a_min_p1_div_2Nij_sumj, r, policy):
    """
        The demand and supply function are unknown,
        :param E: the graph
        :param num_customer: number of customer types
        :param num_server: number of server types
        :param opt_value: optimal profit
        :param min_p_c: minimum price for the customer side
        :param max_p_c: maximum price for the customer side
        :param min_p_s: minimum price for the server side
        :param max_p_s: maximum price for the server side
        :param num_sim: number of simulation

        :return profit_regret: profit regret
        :return cum_queue: total queue length cumulated over time
        :return max_queue: maximum queue length
    """

    results_parallel = Parallel(n_jobs=-1, verbose=1)(
        delayed(alg_learning_single_simu)(num_customer, num_server, E,
                                          policy,
                                          min_p_c, max_p_c, min_p_s, max_p_s,
                                          opt_value,
                                          a_min, a_min_p1_div_2Nij, a_min_p1_div_2Nij_sumi, a_min_p1_div_2Nij_sumj, r)
        for _ in range(num_sim))

    profit_regret, cum_queue, max_queue, k_array = map(
        lambda i: np.stack([result[i] for result in results_parallel], axis=1), range(4))

    return profit_regret, cum_queue, max_queue, k_array


def projection(E, num_customer, num_server, delta, data_point,
               a_min, a_min_p1_div_2Nij, a_min_p1_div_2Nij_sumi, a_min_p1_div_2Nij_sumj, r):
    """
    Solve multiple projection problems with inequality constraints.

    Parameters:
        ...

    Returns:
    - solutions: Array of shape (batch_size, n) containing the solutions.
    """
    assert delta < r, "delta should be smaller than r"
    E_list = [(i, j) for i in range(num_customer) for j in range(num_server) if E[i, j] == 1]
    x = {(i, j): cp.Variable(nonneg=True) for i, j in E_list}

    obj = cp.sum([cp.square(x[i, j] - data_point[i, j]) for (i, j) in E_list])

    # Define the constraints
    constraints = [x[i, j] - a_min_p1_div_2Nij[i, j] >= -(1 - delta / r) * a_min_p1_div_2Nij[i, j] for (i, j) in E_list]
    for i in range(num_customer):
        constraints.append(cp.sum([x[i, j] - a_min_p1_div_2Nij[i, j] for j in range(num_server) if (i, j) in E_list])
                           <= (1 - delta / r) * (1 - a_min_p1_div_2Nij_sumj[i]))
        constraints.append(cp.sum([x[i, j] - a_min_p1_div_2Nij[i, j] for j in range(num_server) if (i, j) in E_list])
                           >= -(1 - delta / r) * (a_min_p1_div_2Nij_sumj[i] - a_min))

    for j in range(num_server):
        constraints.append(cp.sum([x[i, j] - a_min_p1_div_2Nij[i, j] for i in range(num_customer) if (i, j) in E_list])
                           <= (1 - delta / r) * (1 - a_min_p1_div_2Nij_sumi[j]))
        constraints.append(cp.sum([x[i, j] - a_min_p1_div_2Nij[i, j] for i in range(num_customer) if (i, j) in E_list])
                           >= -(1 - delta / r) * (a_min_p1_div_2Nij_sumi[j] - a_min))

    # Define the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)

    # Solve the problem
    result = prob.solve()

    x_array = np.zeros((num_customer, num_server))

    # Print the results
    # print("Optimal value:", prob.value)
    # print("Optimal solution:")
    for (i, j), var in x.items():
        x_array[i, j] = var.value
        # print(f"x[{i},{j}] = {var.value:.4f}")

    return x_array


def alg_matching_lqf(E, queue_len_c, queue_len_s, arrivals_c, arrivals_s):
    """
    Longest Queue First Matching algorithm, which is also MaxWeight Matching algorithm in this case
    In-place adjustment
    :param E: the graph
    :param queue_len_c: queue lengths before the matching for the customer side
    :param queue_len_s: queue lengths before the matching for the server side
    :param arrivals_c: new arrivals at the customer side
    :param arrivals_s: new arrivals at the server side
    """
    num_sim = arrivals_c.shape[1]
    num_customer = arrivals_c.shape[0]
    num_server = arrivals_s.shape[0]
    for i in range(num_customer):
        queue_len_c[i, arrivals_c[i, :] == 1] += 1
        connected = np.array([j for j in range(num_server) if E[i, j] == 1])
        idx = connected[
            np.argmax(queue_len_s[connected, :], axis=0)]  # queue idx of the largest queue length, shape (num_sim,)
        bool_idx = queue_len_s[
                       idx, np.arange(num_sim)] > 0  # whether the largest queue length is empty, shape (num_sim,)
        bool_idx = bool_idx & (arrivals_c[i, :] == 1)  # nonempty and there is arrival, shape (num_sim,)
        idx_nonempty = np.arange(num_sim)[bool_idx]  # sim idx of the nonempty queues, shape (N,), where N <= num_sims
        queue_len_s[idx[bool_idx], idx_nonempty] -= 1
        queue_len_c[i, bool_idx] -= 1

    for j in range(num_server):
        queue_len_s[j, arrivals_s[j, :] == 1] += 1
        connected = np.array([i for i in range(num_customer) if E[i, j] == 1])
        idx = connected[
            np.argmax(queue_len_c[connected, :], axis=0)]  # queue idx of the largest queue length, shape (num_sim,)
        bool_idx = queue_len_c[
                       idx, np.arange(num_sim)] > 0  # whether the largest queue length is empty, shape (num_sim,)
        bool_idx = bool_idx & (arrivals_s[j, :] == 1)  # nonempty and there is arrival, shape (num_sim,)
        idx_nonempty = np.arange(num_sim)[bool_idx]  # sim idx of the nonempty queues, shape (N,), where N <= num_sims
        queue_len_c[idx[bool_idx], idx_nonempty] -= 1
        queue_len_s[j, bool_idx] -= 1

    return


def main():
    # System parameters
    num_customer = 1  # Number of customer types
    num_server = 1  # Number of server types
    num_sim = args.num_sim
    np.random.seed(42)

    # Generate the compatibility matrix
    if num_customer == num_server and num_customer >= 3:
        E = np.zeros((num_customer, num_server), dtype=int)
        for i in range(num_customer):
            for j in range(num_server):
                for k in range(2):
                    if i == j + k:
                        E[i, j] = 1
                    if i == max(j + k - num_server, 0):
                        E[i, j] = 1
    elif num_customer == num_server and num_customer == 1:
        E = np.ones((1, 1), dtype=int)

    Nij = np.zeros((num_customer, num_server), dtype=int)
    for i in range(num_customer):
        for j in range(num_server):
            Nij[i, j] = max(np.sum(E[i, :]), np.sum(E[:, j]))

    a_min = args.a_min
    a_min_p1_div_2Nij = (a_min + 1) / Nij / 2
    a_min_p1_div_2Nij_sumi = np.sum(a_min_p1_div_2Nij, axis=0)
    a_min_p1_div_2Nij_sumj = np.sum(a_min_p1_div_2Nij, axis=1)
    r = min(np.min(a_min_p1_div_2Nij),
            np.min((1 - a_min_p1_div_2Nij_sumj) / np.sum(E, axis=1)),
            np.min((1 - a_min_p1_div_2Nij_sumi) / np.sum(E, axis=0)),
            np.min((a_min_p1_div_2Nij_sumj - a_min) / np.sum(E, axis=1)),
            np.min((a_min_p1_div_2Nij_sumi - a_min) / np.sum(E, axis=0)))

    min_p_c = inv_demand_curve(1)  # minimum price for customer queue
    max_p_c = inv_demand_curve(0)  # maximum price for customer queue

    min_p_s = inv_supply_curve(0)  # minimum price for server queue
    max_p_s = inv_supply_curve(1)  # maximum price for server queue

    x_array, opt_value, arr_c_array, arr_s_array = solve_fluid_model(E, num_customer, num_server)

    if args.policy == "known_two_price":
        profit_regret, cum_queue, max_queue = alg_known(E, num_customer, num_server, opt_value, arr_c_array,
                                                        arr_s_array,
                                                        min_p_c, max_p_c, min_p_s, max_p_s,
                                                        num_sim,
                                                        args.policy)
        avg_queue = cum_queue / np.reshape(np.arange(T) + 1, (-1, 1))
    elif args.policy == "learn_threshold" or args.policy == "learn_two_price_threshold":
        profit_regret, cum_queue, max_queue, k_array = alg_learning(E, num_customer, num_server, opt_value,
                                                                    min_p_c, max_p_c, min_p_s, max_p_s,
                                                                    num_sim,
                                                                    a_min, a_min_p1_div_2Nij,
                                                                    a_min_p1_div_2Nij_sumi, a_min_p1_div_2Nij_sumj, r,
                                                                    args.policy)  # in-place adjustment
        avg_queue = cum_queue / np.reshape(np.arange(T) + 1, (-1, 1))

    print(profit_regret[-1, :])

    print(profit_regret[-1, :].mean())

    if args.policy == "learn_threshold" or args.policy == "learn_two_price_threshold":
        print(k_array)

    np.save(PATH_PROFIT, profit_regret)
    np.save(PATH_AVG_QUEUE, avg_queue)
    np.save(PATH_MAX_QUEUE, max_queue)

    print("T=" + str(T) + " finished")


if __name__ == "__main__":
    main()
