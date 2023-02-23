from algorithms import *
import multiprocess
from multiprocess import Pool
import networkx as nx
from math import log

from itertools import product
import os


def create_problem(Network, Agents, T, N, K, param):
    # param: (mp, n_gossip, gamma, p, n_repeat)
    mp, _, gamma, _, _ = param

    # set gamma and MP protocols in Agents
    for agent in Agents:
        agent.gamma = gamma
        agent.mp = mp
        agent.history = deque(maxlen=gamma*len(Network))

    # create problem instance
    return Problem(Network, Agents, T, N, K, param)


def main_parallel(Network, Agents_, T, N, K, mps, n_gossips, gammas, ps, n_repeats, path):
    if len(gammas) == 1 and len(ps) > 1:
        exp_type = "vary_p"
    elif len(ps) == 1 and len(gammas) > 1:
        exp_type = "vary_gamma"
    elif len(ps) == 1 and len(gammas) == 1:
        exp_type = "vary_t"
    else:
        raise ValueError("Are we fixing p or gamma?")

    # n_gossip, mp, gamma, p, repeat
    params = list(product(mps, n_gossips, gammas, ps, range(n_repeats)))
    # run experiment only once for baseline
    params = [item for item in params if item[0] != "baseline" or (item[0] == "baseline" and item[1] is None)]
    # remove Hitting+Gossiping
    params = [item for item in params if "Hitting" not in item[0] or item[1] is None]

    def F(param):
        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        Agents = deepcopy(Agents_)
        problem = create_problem(Network, Agents, T, N, K, param)
        result = run_ucb(problem, param[-2])

        # print process id
        print(f"Finished: {multiprocess.process.current_process()}, {param}")

        return result

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

    original_edges = list(Network.edges())
    # partial_param = (n_gossip, mp)
    def f(partial_param):
        total_regret = np.zeros((n_repeats, length))
        total_communication = np.zeros((n_repeats, length))
        total_messages_1 = np.zeros((n_repeats, length))
        total_messages_2 = np.zeros((n_repeats, length))

        for repeat, i in product(range(n_repeats), range(length)):
            if exp_type == "vary_t":
                idx = params.index(partial_param + (gamma, p, repeat))
                total_regret[repeat][i] = everything[idx][0][i]
                total_communication[repeat][i] = everything[idx][1][i]
                total_messages_1[repeat][i] = everything[idx][2][original_edges.index((7, 14))][i]
                total_messages_2[repeat][i] = everything[idx][2][original_edges.index((8, 14))][i]
            else:
                if exp_type == "vary_p":
                    idx = params.index(partial_param + (gamma, l[i], repeat))
                elif exp_type == "vary_gamma":
                    idx = params.index(partial_param + (l[i], p, repeat))
                else:
                    raise ValueError("Are we fixing p or gamma?")
                total_regret[repeat][i] = everything[idx][0][-1]
                total_communication[repeat][i] = everything[idx][1][-1]

        return total_regret, total_communication, total_messages_1, total_messages_2

    # collect datas in parallel
    partial_params = list(product(mps, n_gossips))
    # run experiment only once for baseline
    partial_params = [item for item in partial_params if
                      item[0] != "baseline" or (item[0] == "baseline" and item[1] is None)]
    # remove Hitting+Gossiping
    partial_params = [item for item in partial_params if "Hitting" not in item[0] or item[1] is None]

    with Pool() as pool:
        finals = pool.map_async(f, partial_params)
        finals = finals.get()

    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []
    final_messages_mean_1, final_messages_std_1 = [], []
    final_messages_mean_2, final_messages_std_2 = [], []
    for total_regret, total_communication, total_message_1, total_message_2 in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))
        final_messages_mean_1.append(np.mean(total_message_1, axis=0))
        final_messages_std_1.append(np.std(total_message_1, axis=0))
        final_messages_mean_2.append(np.mean(total_message_2, axis=0))
        final_messages_std_2.append(np.std(total_message_2, axis=0))

    if exp_type == "vary_p":
        title_regret = f"Final Regret ({Network.name}, gamma={gamma})"
        fname_regret = f"{path}/Regret_final_p_gamma={gamma}_{Network.name}"

        title_communication = f"Final Communication ({Network.name}, gamma={gamma})"
        fname_communication = f"{path}/Communication_final_p_gamma={gamma}_{Network.name}"
    elif exp_type == "vary_gamma":
        title_regret = f"Final Regret ({Network.name}, p={p})"
        fname_regret = f"{path}/Regret_final_gamma_p={p}_{Network.name}"

        title_communication = f"Final Communication ({Network.name}, p={p})"
        fname_communication = f"{path}/Communication_final_gamma_p={p}_{Network.name}"
    elif exp_type == "vary_t":
        title_regret = f"Final Regret ({Network.name}, p={p}, gamma={gamma})"
        fname_regret = f"{path}/Regret_final_t_p={p}_gamma={gamma}_{Network.name}"

        title_communication = f"Final Communication ({Network.name}, p={p}, gamma={gamma})"
        fname_communication = f"{path}/Communication_final_t_p={p}_gamma={gamma}_{Network.name}"

        title_message_1 = f"# of Messages in Bottleneck Edge (7, 14) ({Network.name}, p={p}, gamma={gamma})"
        fname_message_1 = f"{path}/Message_final_t_(7,14)_p={p}_gamma={gamma}_{Network.name}"

        title_message_2 = f"# of Messages in Bottleneck Edge (8, 14) ({Network.name}, p={p}, gamma={gamma})"
        fname_message_2 = f"{path}/Message_final_t_(8,14)_p={p}_gamma={gamma}_{Network.name}"
    else:
        raise ValueError("Are we fixing p or gamma?")

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting regret and communication
    legends = [f"{mp} (n_gossip={n_gossip})" for mp, n_gossip in partial_params]
    if exp_type == "vary_gamma":
        plot = plot_final_discrete
    else:
        plot = plot_final

    plot(np.array(final_regrets_mean), np.array(final_regrets_std), l,
         title_regret, x_label, legends, f"{fname_regret}.pdf")
    plot(np.array(final_communications_mean), np.array(final_communications_std), l,
         title_communication, x_label, legends, f"{fname_communication}.pdf")
    if exp_type == "vary_t":
        plot(np.array(final_messages_mean_1), np.array(final_messages_std_1), l,
             title_message_1, x_label, legends, f"{fname_message_1}.pdf")
        plot(np.array(final_messages_mean_2), np.array(final_messages_std_2), l,
             title_message_2, x_label, legends, f"{fname_message_2}.pdf")


if __name__ == '__main__':
    num_clusters = 2  # for SBM
    size_cluster = 10
    N = size_cluster * num_clusters  # number of agents
    er_p = 2 * log(N) / N
    # er_p = 1.01 * log(N) / N  # for large instances

    # create communication networks
    Networks = {}
    if not os.path.exists("results/networks"):
        os.makedirs("results/networks")

    ## Erodos-Renyi
    # if the graph is disconnected, keep trying other seeds until the graph is connected.
    u = 2023
    while not nx.is_connected(nx.erdos_renyi_graph(N, er_p, seed=u)):
        u += 1
    Network_ER = nx.erdos_renyi_graph(N, er_p, seed=u)
    Network_ER.name = f"ER_{er_p}"
    pos_ER = nx.spring_layout(Network_ER)
    Networks['ER'] = (Network_ER, pos_ER)
    plot_network(Network_ER, pos_ER)

    ## Barabasi-Albert
    m = 5
    Network_BA = nx.barabasi_albert_graph(N, m, seed=2023)
    Network_BA.name = f"BA_{m}"
    pos_BA = nx.spring_layout(Network_BA)
    Networks['BA'] = (Network_BA, pos_BA)
    plot_network(Network_BA, pos_BA)

    ## Binary SBM
    sbm_p, sbm_q = 2 * er_p, 0.01
    # sbm_p, sbm_q = 3 * er_p, 0.001    # for large instances
    Network_SBM = nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=2023)
    Network_SBM.name = f"SBM_{sbm_p}_{sbm_q}"
    pos_SBM = nx.spring_layout(Network_SBM)
    Networks['SBM'] = (Network_SBM, pos_SBM)
    plot_network(Network_SBM, pos_SBM)

    ## Star Graph
    Network_Star = nx.star_graph(N-1)
    Network_Star.name = f"Star"
    pos_Star = nx.spring_layout(Network_Star)
    Networks['Star'] = (Network_Star, pos_Star)
    plot_network(Network_Star, pos_Star)

    ## Cycle(Ring) Graph
    Network_Cycle = nx.cycle_graph(N)
    Network_Cycle.name = f"Cycle"
    pos_Cycle = nx.spring_layout(Network_Cycle)
    Networks['Cycle'] = (Network_Cycle, pos_Cycle)
    plot_network(Network_Cycle, pos_Cycle)

    ## Path Graph
    Network_Path = nx.path_graph(N)
    Network_Path.name = f"Path"
    pos_Path = nx.spring_layout(Network_Path)
    Networks['Path'] = (Network_Path, pos_Path)
    plot_network(Network_Path, pos_Path)


    # T = int(1e3)  # number of iterations
    T = int(1e4)  # number of iterations    # for path, cycle, star
    K = 20  # total number of arms
    k = 10  # number of arms per agent

    # arm set and their mean rewards
    if not os.path.exists("results/means.npz"):
        tmp = np.random.uniform(size=K)
        np.savez("results/means.npz", tmp=tmp)
    else:
        tmp = np.load("results/means.npz")
        tmp = tmp['tmp']
    # tmp = [0.5 * int(k == K-1) + 0.5 for k in range(K)]
    # tmp = [np.random.uniform() * int(k != K-1) + int(k == K-1) for k in range(K)]
    reward_avgs = {a: tmp[a] for a in range(K)}
    total_arm_set = list(range(K))

    # distributing the arms uniformly at random, until the union over all agents is the whole set
    if not os.path.exists("results/arm_sets.npz"):
        while True:
            arm_sets = [set(np.random.choice(total_arm_set, size=k, replace=False)) for _ in range(N)]

            if set(range(K)) == set.union(*arm_sets):
                arm_sets = [list(item) for item in arm_sets]
                break
        np.savez("results/arm_sets.npz", arm_sets=arm_sets)
    else:
        arm_sets = np.load("results/arm_sets.npz")
        arm_sets = arm_sets['arm_sets']

    # experiments
    for RG_model in ['ER', 'BA', 'SBM', 'Path', 'Cycle', 'Star']:
        for bandwidth in [""]:
        # for bandwidth in ["", "-bandwidth"]:
            print(f"{bandwidth}, {RG_model}; N={N},K={K},k={k},T={T}")
            path = f"results/heterogeneous_K={K}{bandwidth}"
            # create paths
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(path + f"/{RG_model}"):
                os.makedirs(path + f"/{RG_model}")
            if not os.path.exists(f"{path}/networks"):
                os.makedirs(f"{path}/networks")

            # load Network
            Network, pos = Networks[RG_model]

            # create agents with k-uniformly distributed arms
            Agents = [Agent(v, arm_sets[v], reward_avgs, Network, 0, 0, K) for v in range(N)]

            # draw network for each arm
            for arm in range(K):
                color_map = []
                for v in range(N):
                    if arm in Agents[v].arm_set:
                        color_map.append('red')
                    else:
                        color_map.append('blue')

                # color the agents corresponding to the arm
                f = plt.figure(1000 * arm)
                plot_network(Network, pos, f"results/networks/{Network.name}_{arm}.pdf", color_map)

            # algorithms for comparison
            mps, n_gossips = ["baseline", f"Flooding{bandwidth}", f"Flooding-Absorption{bandwidth}"], [None, 1]

            # Experiment #1.1 Comparing regrets (over iteration t)
            gammas = [1]
            ps = [1.0]
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()

            gammas = [2]
            ps = [1.0]
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()

            gammas = [3]
            ps = [1.0]
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()

            # Experiment #1.2 Effect of varying p
            # p: probability that a message is *not* lost
            ps = list(np.linspace(0, 1.0, num=21))
            gammas = [2]  # number of rounds for message passing
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()

            # Experiment #1.3 Effect of gamma, under perfect communication
            gammas = [1, 2, 3, 4]  # max number of rounds for message passing
            ps = [1.0]
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()
