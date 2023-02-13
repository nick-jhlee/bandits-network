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

    # print(F(params[0]))
    # raise ValueError("stop")

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

    # partial_param = (n_gossip, mp)
    def f(partial_param):
        total_regret = np.zeros((n_repeats, length))
        total_communication = np.zeros((n_repeats, length))
        total_messages = np.zeros((len(Network.edges()), T))

        for repeat, i in product(range(n_repeats), range(length)):
            if exp_type == "vary_t":
                idx = params.index(partial_param + (gamma, p, repeat))
                total_regret[repeat][i] = everything[idx][0][i]
                total_communication[repeat][i] = everything[idx][1][i]
                if i == 0:  # update only once per update
                    total_messages += everything[idx][2]
            else:
                if exp_type == "vary_p":
                    idx = params.index(partial_param + (gamma, l[i], repeat))
                elif exp_type == "vary_gamma":
                    idx = params.index(partial_param + (l[i], p, repeat))
                else:
                    raise ValueError("Are we fixing p or gamma?")
                total_regret[repeat][i] = everything[idx][0][-1]
                total_communication[repeat][i] = everything[idx][1][-1]

        return total_regret, total_communication, total_messages / n_repeats

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
    final_messages_means = []
    for total_regret, total_communication, total_messages in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))
        final_messages_means.append(total_messages)

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
    else:
        raise ValueError("Are we fixing p or gamma?")

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting regret and communication
    legends = [f"{mp} (n_gossip={n_gossip})" for mp, n_gossip in partial_params]
    plot_final(np.array(final_regrets_mean), np.array(final_regrets_std), l,
               title_regret, x_label, legends, f"{fname_regret}.pdf")

    plot_final(np.array(final_communications_mean), np.array(final_communications_std), l,
               title_communication, x_label, legends, f"{fname_communication}.pdf")

    # plotting number of messages passed around, per edge
    if exp_type == "vary_t" and n_gossips == [None]:
        original_edges = list(Network.edges())
        for mp_idx, total_messages in enumerate(final_messages_means):
            fig, ax = plt.subplots()
            clrs = sns.color_palette("husl", len(original_edges))
            xs = range(T)
            mp = mps[mp_idx]

            with sns.axes_style("darkgrid"):
                for i, color in enumerate(clrs):
                    ax.plot(xs, total_messages[i], label=f"{original_edges[i]}", c=color)
                    # ax.fill_between(xs, final_means[i] - final_stds[i], final_means[i] + final_stds[i],
                    #                 alpha=0.3, facecolor=color)

                ax.set_title(f"[{mp}] Number of messages per edge (gamma={gamma}, p={p}, n_gossip=None)")
                ax.set_xlabel("t")
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(f"{path}/Messages_{mp}_p={p}_gamma={gamma}_{Network.name}.pdf", dpi=1200,
                            bbox_inches='tight')
                plt.close()
                # plt.show()


if __name__ == '__main__':
    num_clusters = 2  # for SBM
    size_cluster = 10
    N = size_cluster * num_clusters  # number of agents
    er_p = 2 * log(N) / N

    # create communication networks
    Networks = {}
    if not os.path.exists("results/networks"):
        os.makedirs("results/networks")

    ## Erodos-Renyi
    Network_ER = nx.erdos_renyi_graph(N, er_p, seed=2023)
    Network_ER.name = f"ER_{er_p}"
    # if the graph is disconnected, keep trying other seeds until the graph is connected.
    u = 1
    while not nx.is_connected(Network_ER):
        Network_ER = nx.erdos_renyi_graph(N, er_p, seed=2023 + u)
        u += 1
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
    Network_SBM = nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=2023)
    Network_SBM.name = f"SBM_{sbm_p}_{sbm_q}"
    pos_SBM = nx.spring_layout(Network_SBM)
    Networks['SBM'] = (Network_SBM, pos_SBM)
    plot_network(Network_SBM, pos_SBM)

    ## Barbell model
    Network_Barbell = nx.barbell_graph(N // 2, 0)
    Network_Barbell.name = "Barbell"
    pos_Barbell = nx.spring_layout(Network_Barbell)
    Networks['Barbell'] = (Network_Barbell, pos_Barbell)
    plot_network(Network_Barbell, pos_Barbell)

    T = int(1e3)  # number of iterations
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
    for RG_model in ['SBM', 'BA', 'ER', 'Barbell']:
        for bandwidth in ["", "-bandwidth"]:
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
                nx.draw_networkx(Network, node_color=color_map, pos=pos, with_labels=True)
                f.savefig(f"results/networks/{Network.name}_{arm}.pdf", bbox_inches='tight')
                plt.close()
                # plt.show()

            # algorithms for comparison
            mps, n_gossips = ["baseline", f"MP{bandwidth}", f"Hitting-MP{bandwidth}"], [None, 1]

            # Experiment #1.1 Comparing regrets (over iteration t)
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
            ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            gammas = [2]  # number of rounds for message passing
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()

            # Experiment #1.3 Effect of gamma, under perfect communication
            gammas = [1, 2, 3, 4]  # max number of rounds for message passing
            ps = [1.0]
            main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, 10, path + f"/{RG_model}")
            plt.clf()
