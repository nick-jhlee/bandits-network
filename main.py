from algorithms import *
import networkx as nx
import random
import multiprocess
from multiprocess import Pool

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


# Thompson atom model-like arm distribution
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


def main_parallel(Network, Agents_, T, N, K, mps, n_gossips, gammas, ps, n_repeats, path):
    fixed_edge = list(Network.edges())[0]

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

        return result  # Group_Regrets, Group_Communications, np.array(Edge_Messages)

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
        # message_absorption_times = []

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
        # return total_regret, total_communication, edge_messages, message_absorption_times

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

    if exp_type == "vary_p":
        title_regret = f"Final Regret ({Network.name}, gamma={gamma})"
        fname_regret = f"{path}/Regret_final_p_gamma={gamma}"

        title_communication = f"Final Communication ({Network.name}, gamma={gamma})"
        fname_communication = f"{path}/Communication_final_p_gamma={gamma}"
    elif exp_type == "vary_gamma":
        title_regret = f"Final Regret ({Network.name}, p={p})"
        fname_regret = f"{path}/Regret_final_gamma_p={p}"

        title_communication = f"Final Communication ({Network.name}, p={p})"
        fname_communication = f"{path}/Communication_final_gamma_p={p}"
    elif exp_type == "vary_t":
        title_regret = f"Final Regret ({Network.name}, p={p}, gamma={gamma})"
        fname_regret = f"{path}/Regret_final_t_p={p}_gamma={gamma}"

        title_communication = f"Final Communication ({Network.name}, p={p}, gamma={gamma})"
        fname_communication = f"{path}/Communication_final_t_p={p}_gamma={gamma}"
        if 'SBM' in Network.name:
            title_message = f"# of Messages in Bottleneck Edge (11,53) ({Network.name}, p={p}, gamma={gamma})"
        else:
            title_message = f"# of Messages in {fixed_edge} ({Network.name}, p={p}, gamma={gamma})"
        fname_message = f"{path}/Message_final_t_p={p}_gamma={gamma}"
    else:
        raise ValueError("Are we fixing p or gamma?")

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)
    np.savez(f"{fname_message}.npz", final_messages_mean, final_messages_std)

    # plotting regret and communication
    legends = [f"{mp} (n_gossip={n_gossip})" for mp, n_gossip in partial_params]
    if exp_type == "vary_gamma":
        plot = plot_final_discrete
    else:
        plot = plot_final

    plot(np.array(final_regrets_mean), np.array(final_regrets_std), l,
         title_regret, x_label, f"{fname_regret}.pdf", legends)
    plot(np.array(final_communications_mean), np.array(final_communications_std), l,
         title_communication, x_label, f"{fname_communication}.pdf", legends)
    if exp_type == "vary_t":
        plot(np.array(final_messages_mean), np.array(final_messages_std), l,
             title_message, x_label, f"{fname_message}.pdf", legends)

    # if exp_type != "vary_gamma":
    #     # plotting histogram of absorption time of messages over time
    #     with open(f"{path}/message_absorption_times_p={p}_gamma={gamma}.json", 'w') as f:
    #         json.dump(final_absorption_times, f)
    #     plt.hist(final_absorption_times, label=mps)
    #     plt.legend()
    #     plt.savefig(f"{path}/message_absorption_times_p={p}_gamma={gamma}.pdf", dpi=1200,
    #                 bbox_inches='tight')


if __name__ == '__main__':
    num_clusters = 4  # for SBM
    size_cluster = 25
    N = size_cluster * num_clusters  # number of agents
    # er_p = 2 * log(N) / N
    er_p = 3 / N  # for large instances

    # create communication networks
    Networks = {}
    diameters = {}
    if not os.path.exists("networks"):
        os.makedirs("networks")

    ## Erodos-Renyi
    # if the graph is disconnected, keep trying other seeds until the graph is connected.
    u = 2023
    while not nx.is_connected(nx.erdos_renyi_graph(N, er_p, seed=u)):
        u += 1
    Network_ER = nx.erdos_renyi_graph(N, er_p, seed=u)
    Network_ER.name = f"ER_{er_p}"
    pos_ER = nx.spring_layout(Network_ER)
    Networks['ER'] = (Network_ER, pos_ER)
    plot_network(Network_ER, pos_ER, parent="networks")
    diameters['ER'] = int(nx.diameter(Network_ER))
    print(list(Network_ER.edges())[0])

    # Barabasi-Albert
    m = 2
    Network_BA = nx.barabasi_albert_graph(N, m, seed=2023)
    Network_BA.name = f"BA_{m}"
    pos_BA = nx.spring_layout(Network_BA)
    Networks['BA'] = (Network_BA, pos_BA)
    plot_network(Network_BA, pos_BA, parent="networks")
    diameters['BA'] = int(nx.diameter(Network_BA))
    print(list(Network_BA.edges())[0])

    ## Binary SBM
    # sbm_p, sbm_q = 2 * er_p, 0.01
    sbm_p, sbm_q = 10 * er_p, 0.003  # for large instances
    u = 2023
    while not nx.is_connected(
            nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=u)):
        u += 1
    Network_SBM = nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=u)
    Network_SBM.name = f"SBM_{sbm_p}_{sbm_q}"
    pos_SBM = nx.spring_layout(Network_SBM)
    Networks['SBM'] = (Network_SBM, pos_SBM)
    plot_network(Network_SBM, pos_SBM, parent="networks")
    diameters['SBM'] = int(nx.diameter(Network_SBM))
    # print(Network_SBM.adj[20])

    print(diameters)

    T = int(1e3)  # number of iterations
    K = 50  # total number of arms
    k = 20  # number of arms per agent
    n_repeats = 10

    # for uniform in ["nonuniform", "uniform"]:
    for uniform in ["uniform"]:
        base_path = f"results-{uniform}_N_{N}_K_{K}_k_{k}"

        # arm set and their mean rewards
        if not os.path.exists(f"{base_path}/means.npz"):
            tmp = np.sort(np.random.uniform(size=K))
            np.savez(f"{base_path}/means.npz", tmp=tmp)
        else:
            tmp = np.load(f"{base_path}/means.npz")
            tmp = tmp['tmp']
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

        # experiments
        # for dynamic in ["", "_dynamic_sparse", "_dynamic_dense", "_dynamic_hybrid"]:
        for dynamic in ["", "_dynamic_dense"]:
            if dynamic == "":
                RGs = ['ER', 'BA', 'SBM']
            else:
                RGs = ['ER']
            for RG_model in RGs:
                print(f"{dynamic} {RG_model}, {uniform}; N={N},K={K},k={k},T={T}")
                path = f"{base_path}/heterogeneous_K={K}{dynamic}"
                # create paths
                if not os.path.exists(f"{base_path}/networks"):
                    os.makedirs(f"{base_path}/networks")
                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(path + f"/{RG_model}"):
                    os.makedirs(path + f"/{RG_model}")

                # load Network
                Network, pos = Networks[RG_model]

                # set arm sets
                if not os.path.exists(f"{path}/{RG_model}/arm_sets.json"):
                    # "uniform" or "nonuniform"
                    if uniform == "uniform":
                        arm_sets = arm_sets_uniform
                    else:
                        # arm_sets = create_nonuniform_arm_sets(Network, K, arm_sets_uniform)
                        arm_sets = create_nonuniform_arm_sets(Network, K, k)
                    with open(f"{path}/{RG_model}/arm_sets.json", "w") as f:
                        json.dump(arm_sets, f)
                else:
                    with open(f"{path}/{RG_model}/arm_sets.json", "r") as f:
                        arm_sets = json.load(f)

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
                    plot_network(Network, pos,
                                 fname=f"results-{uniform}_N_{N}_K_{K}_k_{k}/networks/{Network.name}_{arm}.pdf",
                                 node_color=color_map)
                    plt.close()

                # Figure 3-5
                mps, n_gossips = [f"baseline{dynamic}", f"Flooding-Absorption{dynamic}", f"Flooding{dynamic}",
                                  f"Flooding-RandomStop{dynamic}-0.5"], [1, None]
                gammas = [4]
                ps = [1.0]
                main_parallel(Network, Agents, T, N, K, mps, n_gossips, gammas, ps, n_repeats, path + f"/{RG_model}")
                plt.clf()
