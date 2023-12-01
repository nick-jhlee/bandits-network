from algorithms import *
import networkx as nx
import random
from math import log

from itertools import product, chain
import os, json


def create_problem(Network, Agents, T, N, K, param):
    # param: (mp, n_gossip, gamma, p, n_repeat)
    mp, _, gamma, _, _ = param

    # set gamma and MP protocols in Agents
    for agent in Agents:
        agent.gamma = gamma
        agent.mp = mp
        agent.history = deque(maxlen=gamma * len(Network))

    # create problem instance
    return Problem(Network, Agents, T, N, K, param)


# mode: "uniform" or "nonuniform"
def create_uniform_arm_sets(N, K, k):
    while True:
        arm_sets = [random.sample(range(K), k) for _ in range(N)]  # uniformly random
        # check if all arms are covered
        if set(range(K)) == set(chain.from_iterable(arm_sets)):
            break
    return arm_sets


## previous ver
# def create_nonuniform_arm_sets(Network, K, arm_sets_uniform):
#     N = Network.number_of_nodes()
#     arm_sets = arm_sets_uniform
#     # prob distribution w.r.t. which the agents to be assigned the globally optimal arm K - 1
#     ps = [Network.degree(v) / (2 * Network.number_of_edges()) for v in range(N)]
#     # make sure that the globally optimal arm K - 1 is in every arm set, initially
#     for arm_set in arm_sets:
#         if K - 1 not in arm_set:
#             arm_set.append(K - 1)
#     # remove K - 1 from agents w.p. (roughly) proportional to the degree
#     remove_agents = np.random.choice(range(N), int(2*N / 3), replace=False, p=ps)
#     for v in remove_agents:
#         arm_sets[v].remove(K - 1)
#     return arm_sets

# Thompson model-like arm distribution
# randomized algorithm that distributes the arms in a maximally separated fashion
def create_nonuniform_arm_sets(Network, K, k, min_dist=5):
    N = Network.number_of_nodes()
    arm_sets = [[] for _ in range(N)]
    Network_power = nx.power(Network, min_dist)
    vertices_not_covered = set(range(N))
    for a in range(K):
        # random maximal independent set, whose vertices are at least min_dist apart
        max_ind_set = nx.maximal_independent_set(Network_power)
        for v in max_ind_set:
            arm_sets[v].append(a)
            if v in vertices_not_covered:
                vertices_not_covered.remove(v)

    while len(vertices_not_covered) > 0:
        v = vertices_not_covered.pop()
        # for a in range(K):
        #     for w in Network.neighbors(v):
        #         if a not in arm_sets[w]:
        #             arm_sets[v].append(a)
        #             if v in vertices_not_covered:
        #                 vertices_not_covered.remove(v)
        arm_sets[v] = np.random.choice(range(K), k, replace=False).tolist()

    return arm_sets


def main_parallel(Network, Agents_, T, N, K, mps, n_gossips, gammas, ps, n_repeats):
    exp_type = "vary_gamma"

    # n_gossip, mp, gamma, p, repeat
    params = list(product(mps, n_gossips, gammas, ps, range(n_repeats)))
    # run experiment only once for baseline
    params = [item for item in params if "baseline" not in item[0] or ("baseline" in item[0] and item[1] is None)]
    # remove Hitting+Gossiping
    params = [item for item in params if "Absorption" not in item[0] or item[1] is None]
    # remove RS+Gossiping
    params = [item for item in params if "RandomStop" not in item[0] or item[1] is None]

    def F(param):
        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        Agents = deepcopy(Agents_)
        problem = create_problem(Network, Agents, T, N, K, param)
        result = run_ucb(problem, param[-2])

        # print process id
        print(f"Finished: {multiprocess.process.current_process()}, {param}")

        return result  # Group_Regrets, Group_Communications, np.array(Edge_Messages), message_absorption_times

    # run the experiments in parallel
    with Pool() as pool:
        everything = pool.map_async(F, params)
        everything = everything.get()

    if exp_type == "vary_p":
        length = len(ps)
        l = ps
        x_label = "p"
        gamma = gammas[0]
    elif exp_type == "vary_gamma":
        length = len(gammas)
        l = gammas
        x_label = "gamma"
        p = ps[0]
    elif exp_type == "vary_t":
        length = T
        l = range(T)
        x_label = "t"
        p = ps[0]
        gamma = gammas[0]
    else:
        raise ValueError("Are we fixing p or gamma?")


    print("Data collection and plotting!")
    # partial_param = (n_gossip, mp)
    def f(partial_param):
        total_regret = np.zeros((n_repeats, length))
        total_communication = np.zeros((n_repeats, length))
        edge_messages = np.zeros((n_repeats, length))

        for repeat, i in product(range(n_repeats), range(length)):
            if exp_type == "vary_t":
                idx = params.index(partial_param + (gamma, p, repeat))
                total_regret[repeat][i] = everything[idx][0][i]
                total_communication[repeat][i] = everything[idx][1][i]
                # if i == 0:
                #     message_absorption_times += everything[idx][3]
                # number of messages passing through the bottleneck edge!
                edge_messages[repeat][i] = everything[idx][2][i]
            else:
                if exp_type == "vary_p":
                    idx = params.index(partial_param + (gamma, l[i], repeat))
                elif exp_type == "vary_gamma":
                    idx = params.index(partial_param + (l[i], p, repeat))
                else:
                    raise ValueError("Are we fixing p or gamma?")
                # final regret and final communication complexity only!
                total_regret[repeat][i] = everything[idx][0][-1]
                total_communication[repeat][i] = everything[idx][1][-1]

        return total_regret, total_communication, edge_messages

    # collect datas in parallel
    partial_params = list(product(mps, n_gossips))
    # run experiment only once for baseline
    partial_params = [item for item in partial_params if
                      "baseline" not in item[0] or ("baseline" in item[0] and item[1] is None)]
    # remove Hitting+Gossiping
    partial_params = [item for item in partial_params if "Absorption" not in item[0] or item[1] is None]
    # remove RS+Gossiping
    partial_params = [item for item in partial_params if "RandomStop" not in item[0] or item[1] is None]

    with Pool() as pool:
        finals = pool.map_async(f, partial_params)
        finals = finals.get()

    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []
    final_messages_mean, final_messages_std = [], []
    for total_regret, total_communication, edge_message in finals:
    # for total_regret, total_communication, edge_message, total_absorption_time in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))
        # final_absorption_times.append(total_absorption_time)
        final_messages_mean.append(np.mean(edge_message, axis=0))
        final_messages_std.append(np.std(edge_message, axis=0))

    return final_regrets_mean, final_regrets_std


if __name__ == '__main__':
    num_clusters = 4  # for SBM
    size_cluster = 25
    N = size_cluster * num_clusters  # number of agents
    er_ps = [1e-2 / N, 1e-1 / N, 1 / N, 2 / N, 3 / N, 4 / N, 5 / N, 6 / N, 7 / N, 8 / N, 9 / N, 10 / N]

    T = int(1e3)  # number of iterations
    K = 50  # total number of arms
    k = 20  # number of arms per agent
    n_repeats = 10

    # create communication networks
    Networks = {}
    if not os.path.exists("deltas/networks"):
        os.makedirs("deltas/networks")

    base_path = f"deltas/results-uniform_N_{N}_K_{K}_k_{k}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # arm set and their mean rewards
    if not os.path.exists(f"{base_path}/means.npz"):
        tmp = np.sort(np.random.uniform(size=K))
        np.savez(f"{base_path}/means.npz", tmp=tmp)
    else:
        tmp = np.load(f"{base_path}/means.npz")
        tmp = tmp['tmp']
    # if not os.path.exists("results-{uniform}/means.npz"):
    #     tmp = np.sort(np.random.uniform(size=K))
    #     np.savez("results-{uniform}/means.npz", tmp=tmp)
    # else:
    #     tmp = np.load("results-{uniform}/means.npz")
    #     tmp = tmp['tmp']
    # tmp = [0.5 * int(k == K-1) + 0.5 for k in range(K)]
    # tmp = [np.random.uniform() * int(k != K-1) + int(k == K-1) for k in range(K)]
    reward_avgs = {a: tmp[a] for a in range(K)}
    total_arm_set = list(range(K))

    # uniform arm distribution
    if not os.path.exists(f"{base_path}/arm_sets_uniform.json"):
        arm_sets_uniform = create_uniform_arm_sets(N, K, k)
        with open(f"{base_path}/arm_sets_uniform.json", "w") as f:
            json.dump(arm_sets_uniform, f)
    else:
        with open(f"{base_path}/arm_sets_uniform.json", "r") as f:
            arm_sets_uniform = json.load(f)

    regret_Flooding_mean, regret_FwA_mean = [], []
    regret_Flooding_std, regret_FwA_std = [], []
    for er_p in er_ps:
        print(er_p)
        ## Erodos-Renyi
        # if the graph is disconnected, keep trying other seeds until the graph is connected.
        u = 2023
        Network_ER = nx.erdos_renyi_graph(N, er_p, seed=u)
        Network_ER.name = f"ER_{er_p}"
        pos_ER = nx.spring_layout(Network_ER)
        Networks[er_p] = (Network_ER, pos_ER)
        plot_network(Network_ER, pos_ER, parent="deltas/networks")

        # experiments
        # create paths
        if not os.path.exists(f"{base_path}/networks"):
            os.makedirs(f"{base_path}/networks")

        # load Network
        Network, pos = Network_ER, pos_ER
        arm_sets = arm_sets_uniform

        # create agents with the distributed arms
        Agents = [Agent(v, arm_sets[v], reward_avgs, Network, 0, 0, K) for v in range(N)]

        # plot arm-specific networks
        for arm in range(K):
            color_map = []
            for v in range(N):
                if arm in Agents[v].arm_set:
                    color_map.append('red')
                else:
                    color_map.append('blue')

            # color the agents corresponding to the arm
            f = plt.figure(1000 * arm)
            plot_network(Network, pos, fname=f"deltas/networks/{Network.name}_{arm}.pdf", node_color=color_map)
            plt.close()


        mps, n_gossips = ["Flooding-Absorption", "Flooding"], [None]
        gammas = [4]
        ps = [1.0]
        means, stds = main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, n_repeats)
        regret_Flooding_mean.append(means[0])
        regret_Flooding_std.append(stds[0])
        regret_FwA_mean.append(means[1])
        regret_FwA_std.append(stds[1])

    regret_Flooding_mean = np.array(regret_Flooding_mean)
    regret_Flooding_std = np.array(regret_Flooding_std)
    regret_FwA_mean = np.array(regret_FwA_mean)
    regret_FwA_std = np.array(regret_FwA_std)

    # save and plot
    np.savez(f"{base_path}/regret_Flooding.npz", regret_Flooding_mean=regret_Flooding_mean, regret_Flooding_std=regret_Flooding_std)
    np.savez(f"{base_path}/regret_FwA.npz", regret_FwA_mean=regret_FwA_mean, regret_FwA_std=regret_FwA_std)

    plt.figure()
    plt.errorbar(er_ps, regret_Flooding_mean[:, -1], yerr=regret_Flooding_std[:, -1], label="Flooding", capsize=5)
    plt.errorbar(er_ps, regret_FwA_mean[:, -1], yerr=regret_FwA_std[:, -1], label="Flooding-Absorption", capsize=5)
    plt.xlabel("er_p")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig(f"{base_path}/regret.pdf")
    plt.show()
    plt.clf()