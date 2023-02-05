from algorithms import *
from multiprocess import Pool
import networkx as nx
from itertools import product
import time
import argparse


def create_problem(T, N, K, Network, param):
    discard, n_gossip, mp, _, gamma = param

    # arm set and their mean rewards
    total_arm_set = list(range(K))
    tmp = np.linspace(0.1, 1.0, num=K)
    reward_avgs = {total_arm_set[i]: tmp[i] for i in range(K)}

    # create agents
    Agents = []
    for i in range(N):
        if i < 4:
            arm_set_i = total_arm_set[5 * i:5 * (i + 1)]
            Agents.append(Agent(i, arm_set_i, reward_avgs, Network, gamma, mp))
        else:
            arm_set_i = np.random.choice(total_arm_set, size=5, replace=False)
            Agents.append(Agent(i, arm_set_i, reward_avgs, Network, gamma, mp))

    # create problem instance
    return Problem(Agents, Network, T, N, mp, n_gossip, discard)


def main_ps_parallel(T, N, K, Network, gamma, discards, n_gossips, mps, ps, repeats=10):
    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []

    # discard, n_gossip, mp, repeats, ps
    params = list(product(discards, n_gossips, mps, range(repeats), ps))

    def F(param):
        # # print process id
        # print(f"{multiprocess.process.current_process()}, {param}")

        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        problem = create_problem(T, N, K, Network, param)

        return run_ucb(problem, param[-1])

    # run the experiments in parallel
    with Pool() as pool:
        everything = pool.map_async(F, params)
        everything = everything.get()

    # collect datas in parallel
    partial_params = list(product(discards, n_gossips, mps))

    def f(partial_param):
        len_ps = len(ps)

        total_regret = np.zeros((repeats, len_ps))
        total_communication = np.zeros((repeats, len_ps))

        for repeat, i_p in product(range(repeats), range(len_ps)):
            idx = params.index(partial_param + (repeat, ps[i_p]))
            total_regret[repeat][i_p] = everything[idx][0]
            total_communication[repeat][i_p] = everything[idx][1]

        return total_regret, total_communication

    with Pool() as pool:
        finals = pool.map_async(f, partial_params)
        finals = finals.get()

    for total_regret, total_communication in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))

    fname_regret = f"heterogeneous/Regret_final_p_gamma={gamma}_{Network.name}"
    fname_communication = f"heterogeneous/Communication_final_p_gamma={gamma}_{Network.name}"

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting
    legends = [f"{mp} (n_gossip={n_gossip}, discard={discard})" for discard, n_gossip, mp in partial_params]
    plot_final(np.array(final_regrets_mean), np.array(final_regrets_std), ps,
               f"Final Regret ({Network.name}, gamma={gamma})", "p",
               legends, f"{fname_regret}.pdf")

    plot_final(np.array(final_communications_mean), np.array(final_communications_std), ps,
               f"Final Communication ({Network.name}, gamma={gamma})", "p",
               legends, f"{fname_communication}.pdf")


def main_gammas_parallel(T, N, K, Network, p, discards, n_gossips, mps, gammas, repeats=10):
    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []

    # discard, n_gossip, mp, repeats, ps
    params = list(product(discards, n_gossips, mps, range(repeats), gammas))

    def F(param):
        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        problem = create_problem(T, N, K, Network, param)

        return run_ucb(problem, p)

    # run the experiments in parallel
    with Pool() as pool:
        everything = pool.map_async(F, params)
        everything = everything.get()

    # collect datas in parallel
    partial_params = list(product(discards, n_gossips, mps))

    def f(partial_param):
        len_gammas = len(gammas)

        total_regret = np.zeros((repeats, len_gammas))
        total_communication = np.zeros((repeats, len_gammas))

        for repeat, i_gamma in product(range(repeats), range(len_gammas)):
            idx = params.index(partial_param + (repeat, gammas[i_gamma]))
            total_regret[repeat][i_gamma] = everything[idx][0]
            total_communication[repeat][i_gamma] = everything[idx][1]

        return total_regret, total_communication

    with Pool() as pool:
        finals = pool.map_async(f, partial_params)
        finals = finals.get()

    for total_regret, total_communication in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))

    fname_regret = f"heterogeneous/Regret_final_gamma_p={p}_{Network.name}"
    fname_communication = f"heterogeneous/Communication_final_gamma_p={p}_{Network.name}"

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting
    legends = [f"{mp} (n_gossip={n_gossip}, discard={discard})" for discard, n_gossip, mp in partial_params]
    plot_final(np.array(final_regrets_mean), np.array(final_regrets_std), gammas,
               f"Final Regret ({Network.name}, p={p})", "gamma",
               legends, f"{fname_regret}.pdf")

    plot_final(np.array(final_communications_mean), np.array(final_communications_std), gammas,
               f"Final Communication ({Network.name}, p={p})", "gamma",
               legends, f"{fname_communication}.pdf")


if __name__ == '__main__':
    T = int(1e5)  # number of iterations
    N, K = 20, 20  # number of agents, total number of arms
    RG_model = 'ER'

    # compared baseline models
    # discards, n_gossips, mps = [False, True], [1, 3, None], ["MP", "Greedy-MP", "HMP"]
    discards, n_gossips, mps = [False], [1, 3, None], ["MP", "Greedy-MP", "HMP"]

    # create communication network
    for er_p in [0.1, 0.8, 0.4]:
        if RG_model == "ER":
            Network = nx.erdos_renyi_graph(N, 0.7, seed=2023)
            # connect the graph, if it's disconnected
            if not nx.is_connected(Network):
                augment_edges = nx.k_edge_augmentation(Network, 1)
                Network.add_edges_from(augment_edges)
        elif RG_model == "BA":
            Network = nx.barabasi_albert_graph(N, 5, seed=2023)
        else:
            raise NotImplementedError(f"{RG_model} not yet implemented")
        Network.name = f"{RG_model}_{er_p}"

        f = plt.figure(100)
        nx.draw_networkx(Network, with_labels=True)
        f.savefig(f"heterogeneous/{RG_model}.pdf", bbox_inches='tight')
        # plt.show()

        # Experiment #1. Effect of p
        ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # probability that a message is *not* discarded!
        gamma = 3  # number of rounds for message passing
        # main_ps(T, N, K, Network, gamma, discards, n_gossips, mps, ps, 3)
        main_ps_parallel(T, N, K, Network, gamma, discards, n_gossips, mps, ps, 20)

        # Experiment #2. Effect of gamma, under perfect communication
        gammas = [1, 2, 3, 4, 5, 6]  # max number of rounds for message passing
        p = 1.0
        main_gammas_parallel(T, N, K, Network, p, discards, n_gossips, mps, gammas, 20)
