from algorithms import *
import multiprocess
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


def main_parallel(Network, Agents, T, N, K, discards, n_gossips, mps, gammas, ps, n_repeats, path):
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
        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        problem = create_problem(Network, Agents, T, N, K, param)
        result = run_ucb(problem, param[-2])

        # print process id
        print(f"Finished: {multiprocess.process.current_process()}, {param}")

        return result
    print(F(params[0]))
    raise ValueError("stop")

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
        fname_regret = f"{path}/Regret_final_p_gamma={gamma}_{Network.name}"

        title_communication = f"Final Communication ({Network.name}, gamma={gamma})"
        fname_communication = f"heterogeneous_K={K}/Communication_final_p_gamma={gamma}_{Network.name}"
    elif exp_type == "vary_gamma":
        title_regret = f"Final Regret ({Network.name}, p={p})"
        fname_regret = f"{path}/Regret_final_gamma_p={p}_{Network.name}"

        title_communication = f"Final Communication ({Network.name}, p={p})"
        fname_communication = f"{path}/Communication_final_gamma_p={p}_{Network.name}"
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
    for K in [40]:  # total number of arms
        print(f"K={K}")
        T = int(5e2)  # number of iterations
        N = 10  # number of agents
        RG_model = 'ER'
        q = 0.01

        path = f"heterogeneous_K={K}"

        # create path
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(f"{path}/networks"):
            os.makedirs(f"{path}/networks")

        # create communication network
        for er_p in [0.1]:
            if RG_model == "ER":
                Network = nx.erdos_renyi_graph(N, er_p, seed=2023)
                # connect the graph, if it's disconnected
                if not nx.is_connected(Network):
                    augment_edges = nx.k_edge_augmentation(Network, 1)
                    Network.add_edges_from(augment_edges)
            elif RG_model == "BA":
                Network = nx.barabasi_albert_graph(N, 5, seed=2023)
            elif RG_model == "SBM":  # binary SBM
                partition_size = N // 2
                Network = nx.random_partition_graph([partition_size for _ in range(2)], 0.8, 0.1, seed=2023)
            else:
                raise NotImplementedError(f"{RG_model} not yet implemented")
            Network.name = f"{RG_model}_{er_p}"

            # # plant a clique with the global optimal arm K-1
            # cliques = list(nx.find_cliques(Network))
            # clique_sizes = [len(clique) for clique in cliques]
            # max_clique = cliques[np.argmax(clique_sizes)]
            max_clique = []

            if not os.path.exists(f"{path}/networks/{RG_model}_{er_p}.pdf"):
                # plot total network
                f = plt.figure(100 * er_p)
                nx.draw_networkx(Network, with_labels=True)
                f.savefig(f"{path}/networks/{RG_model}_{er_p}.pdf", bbox_inches='tight')
                # plt.show()

            # arm set and their mean rewards
            tmp = np.linspace(0.1, 1.0, num=K)
            # tmp = [0.5 * int(k == K-1) + 0.5 for k in range(K)]
            reward_avgs = {k: tmp[k] for k in range(K)}
            total_arm_set = list(range(K))
            # fixed list of arm sets
            num_arm_sets = K // 5
            arm_sets = [total_arm_set[5 * i:5 * (i + 1)] for i in range(num_arm_sets)]

            # create agents by distributing arms
            Agents = []
            if RG_model != "SBM":
                for i in range(N):
                    if i < K // 5:
                        arm_set_i = total_arm_set[5 * i:5 * (i + 1)]
                    else:
                        arm_set_i = list(np.random.choice(total_arm_set, size=5, replace=False))
                    if i in max_clique and K - 1 not in arm_set_i:
                        arm_set_i.append(K - 1)
                    # arm_set_i = arm_sets[i % num_arm_sets]
                    Agents.append(Agent(i, arm_set_i, reward_avgs, Network, 0, 0, q, K))
            else:
                for i, partition in enumerate(Network.graph['partition']):
                    for v in partition:
                        # best local arm is the same for all agents!
                        arm_set_v = [total_arm_set[K - 1]] + list(
                            np.random.choice(total_arm_set[:K - 1 - i], size=5, replace=False))
                        Agents.append(Agent(v, arm_set_v, reward_avgs, Network, 0, 0, q, K))

            # draw network for each arm
            pos = nx.spring_layout(Network)
            for arm in range(K):
                color_map = []
                for i in range(N):
                    if arm in Agents[i].arm_set:
                        color_map.append('red')
                    else:
                        color_map.append('blue')

                # color the agents corresponding to the arm
                f = plt.figure(1000 * arm)
                nx.draw_networkx(Network, node_color=color_map, pos=pos, with_labels=True)
                f.savefig(f"{path}/networks/{RG_model}_{er_p}_{arm}.pdf", bbox_inches='tight')
                plt.close()
                # plt.show()

            # compared baseline models
            # discards, n_gossips, mps = [False, True], [1, 3, None], ["MP", "Greedy-MP", "Hitting-MP"]
            # discards, n_gossips, mps = [False], [None], ["MP", "Hitting-MP", "corrupt-MP", "corrupt-Hitting-MP"]
            discards, n_gossips, mps = [False], [None], ["baseline", "MP", "Hitting-MP", "bandwidth-MP", "bandwidth-Hitting-MP"]

            # # Experiment #1. Effect of varying p
            # # p: probability that a message is *not* discarded, per link
            # ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # gammas = [2]  # number of rounds for message passing
            # main_parallel(Network, Agents, T, N, K, discards, n_gossips, mps, gammas, ps, 10, path)

            # Experiment #2. Effect of gamma, under perfect communication
            # gammas = [1, 2, 3]  # max number of rounds for message passing
            gammas = [3]  # max number of rounds for message passing
            ps = [1.0]
            main_parallel(Network, Agents, T, N, K, discards, n_gossips, mps, gammas, ps, 10, path)

            plt.clf()
