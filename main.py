from algorithms import *
from multiprocess import Pool
import networkx as nx

from itertools import product
import os
import argparse


def create_problem(Network, Agents, T, N, K, param):
    # param: (discard, n_gossip, mp, gamma, p, n_repeat)
    _, _, mp, gamma, _, _ = param

    # set gamma and MP protocols in Agents
    for agent in Agents:
        agent.gamma = gamma
        agent.mp = mp

    # create problem instance
    return Problem(Network, Agents, T, N, K, param)


def main_parallel(Network, Agents, T, N, K, discards, n_gossips, mps, gammas, ps, n_repeats):
    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []

    if len(gammas) == 1:
        exp_type = "vary_p"
    elif len(ps) == 1:
        exp_type = "vary_gamma"
    else:
        raise ValueError("Are we fixing p or gamma?")

    # discard, n_gossip, mp, gamma, p, repeat
    params = list(product(discards, n_gossips, mps, gammas, ps, range(n_repeats)))

    def F(param):
        # # print process id
        # print(f"{multiprocess.process.current_process()}, {param}")

        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        problem = create_problem(Network, Agents, T, N, K, param)

        return run_ucb(problem, param[-2])

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
    else:
        raise ValueError("Are we fixing p or gamma?")

    # partial_param = (discard, n_gossip, mp)
    def f(partial_param):
        total_regret = np.zeros((n_repeats, length))
        total_communication = np.zeros((n_repeats, length))

        for repeat, i in product(range(n_repeats), range(length)):
            if exp_type == "vary_p":
                idx = params.index(partial_param + (gamma, l[i], repeat))
            elif exp_type == "vary_gamma":
                idx = params.index(partial_param + (l[i], p, repeat))
            else:
                raise ValueError("Are we fixing p or gamma?")
            total_regret[repeat][i] = everything[idx][0]
            total_communication[repeat][i] = everything[idx][1]

        return total_regret, total_communication

    # collect datas in parallel
    partial_params = list(product(discards, n_gossips, mps))
    with Pool() as pool:
        finals = pool.map_async(f, partial_params)
        finals = finals.get()

    for total_regret, total_communication in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))

    if exp_type == "vary_p":
        title_regret = f"Final Regret ({Network.name}, gamma={gamma})"
        fname_regret = f"heterogeneous/Regret_final_p_gamma={gamma}_{Network.name}"
        title_communication = f"Final Communication ({Network.name}, gamma={gamma})"
        fname_communication = f"heterogeneous/Communication_final_p_gamma={gamma}_{Network.name}"
    elif exp_type == "vary_gamma":
        title_regret = f"Final Regret ({Network.name}, p={p})"
        fname_regret = f"heterogeneous/Regret_final_gamma_p={p}_{Network.name}"
        title_communication = f"Final Regret ({Network.name}, p={p})"
        fname_communication = f"heterogeneous/Communication_final_gamma_p={p}_{Network.name}"
    else:
        raise ValueError("Are we fixing p or gamma?")

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting
    legends = [f"{mp} (n_gossip={n_gossip}, discard={discard})" for discard, n_gossip, mp in partial_params]
    plot_final(np.array(final_regrets_mean), np.array(final_regrets_std), l,
               title_regret, x_label, legends, f"{fname_regret}.pdf")

    plot_final(np.array(final_communications_mean), np.array(final_communications_std), l,
               title_communication, x_label, legends, f"{fname_communication}.pdf")


if __name__ == '__main__':
    T = int(1e3)  # number of iterations for each run of bandit
    N, K = 20, 40  # number of agents, total number of arms
    RG_model = 'ER'

    # compared baseline models
    # discards, n_gossips, mps = [False, True], [1, 3, None], ["MP", "Greedy-MP", "Hitting-MP"]
    discards, n_gossips, mps = [False], [None], ["MP", "Greedy-MP", "Hitting-MP"]

    # create communication network
    for er_p in [0.8, 0.1, 0.4]:
        if RG_model == "ER":
            Network = nx.erdos_renyi_graph(N, er_p, seed=2023)
            # connect the graph, if it's disconnected
            if not nx.is_connected(Network):
                augment_edges = nx.k_edge_augmentation(Network, 1)
                Network.add_edges_from(augment_edges)
        elif RG_model == "BA":
            Network = nx.barabasi_albert_graph(N, 5, seed=2023)
        else:
            raise NotImplementedError(f"{RG_model} not yet implemented")
        Network.name = f"{RG_model}_{er_p}"

        # plot total network
        f = plt.figure(100 * er_p)
        nx.draw_networkx(Network, with_labels=True)
        f.savefig(f"heterogeneous/networks/{RG_model}_{er_p}.pdf", bbox_inches='tight')
        # plt.show()

        # arm set and their mean rewards
        total_arm_set = list(range(K))
        tmp = np.linspace(0.1, 1.0, num=K)
        reward_avgs = {total_arm_set[i]: tmp[i] for i in range(K)}

        # create agents by distributing arms
        Agents = []
        for i in range(N):
            if i < K // 5:
                arm_set_i = total_arm_set[5 * i:5 * (i + 1)]
                Agents.append(Agent(i, arm_set_i, reward_avgs, Network, 0, 0))
            else:
                arm_set_i = np.random.choice(total_arm_set, size=5, replace=False)
                Agents.append(Agent(i, arm_set_i, reward_avgs, Network, 0, 0))

        # draw network for each arm
        for arm in range(K):
            color_map = []
            for i in range(N):
                if arm in Agents[i].arm_set:
                    color_map.append('red')
                else:
                    color_map.append('blue')

            # color the agents corresponding to the arm
            f = plt.figure(1000 * arm)
            nx.draw_networkx(Network, node_color=color_map, with_labels=True)
            f.savefig(f"heterogeneous/networks/{RG_model}_{er_p}_{arm}.pdf", bbox_inches='tight')
            # plt.show()

        # Experiment #1. Effect of varying p
        # ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # probability that a message is *not* discarded!
        ps = [0.1, 0.5, 0.9, 1.0]  # probability that a message is *not* discarded!
        gammas = [3]  # number of rounds for message passing
        main_parallel(Network, Agents, T, N, K, discards, n_gossips, mps, gammas, ps, 10)

        # Experiment #2. Effect of gamma, under perfect communication
        gammas = [1, 3, 5, 6]  # max number of rounds for message passing
        ps = [1.0]
        main_parallel(Network, Agents, T, N, K, discards, n_gossips, mps, gammas, ps, 10)
