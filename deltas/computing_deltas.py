from utils_delta import *
from multiprocessing import Pool
import os
import networkx as nx
import json
import matplotlib.pyplot as plt

## same seeding for the networks
num_clusters = 4  # for SBM
size_cluster = 25
N = size_cluster * num_clusters  # number of agents
# er_p = 2 * log(N) / N

# er_p = 3 / N  # for large instances
# create communication networks
Networks = []
Thetas, Thetas_FwA, deltas = [], [], []
diameters = {}
if not os.path.exists("networks"):
    os.makedirs("networks")

K = 50
k = 20
base_path = f"results-uniform_N_{N}_K_{K}_k_{k}"
tmp = np.load(f"../{base_path}/means.npz")
tmp = tmp['tmp']
reward_avgs = {a: tmp[a] for a in range(K)}
total_arm_set = list(range(K))
with open(f"../{base_path}/arm_sets_uniform.json", "r") as f:
    arm_sets = json.load(f)

# compute some stuff first
Deltas = np.zeros((K, 100))
for v in range(N):
    arm_set_v = arm_sets[v]
    for a in arm_set_v:
        Deltas[a][v] = max([reward_avgs[a_] for a_ in arm_set_v]) - reward_avgs[a]

# tilde{Delta]_a as in Yang et al., INFOCOM 2022
Deltas_tilde = []
for a in range(K):
    try:
        Deltas_tilde.append(min(Deltas[a][np.nonzero(Deltas[a])[0]]))
    except:
        Deltas_tilde.append(0)


gamma = 4
er_ps = [1e-2 / N, 1e-1 / N, 1 / N, 2 / N, 3 / N, 4 / N, 5 / N, 6 / N, 7 / N, 8 / N, 9 / N, 10 / N]
for er_p in er_ps:
    print(er_p)
    ## Erodos-Renyi
    # if the graph is disconnected, keep trying other seeds until the graph is connected.
    u = 2023
    # while not nx.is_connected(nx.erdos_renyi_graph(N, er_p, seed=u)):
    #     u += 1
    Network_ER = nx.erdos_renyi_graph(N, er_p, seed=u)
    Network_ER.name = f"ER_{er_p}"
    Networks.append(Network_ER)
    # diameters[er_p] = int(nx.diameter(Network_ER))

    thetas, thetas_FwA, delta = compute_delta(Network_ER, arm_sets, K, gamma, Deltas, Deltas_tilde)

    Thetas.append(thetas)
    Thetas_FwA.append(thetas_FwA)
    deltas.append(delta)

    # # Barabasi-Albert
    # m = 2
    # Network_BA = nx.barabasi_albert_graph(N, m, seed=2023)
    # Network_BA.name = f"BA_{m}"
    # Networks.append(Network_BA)
    # diameters['BA'] = int(nx.diameter(Network_BA))
    #
    # ## Binary SBM
    # # sbm_p, sbm_q = 2 * er_p, 0.01
    # sbm_p, sbm_q = 10 * er_p, 0.003  # for large instances
    # u = 2023
    # while not nx.is_connected(
    #         nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=u)):
    #     u += 1
    # Network_SBM = nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=u)
    # Network_SBM.name = f"SBM_{sbm_p}_{sbm_q}"
    # Networks.append(Network_SBM)
    # diameters['SBM'] = int(nx.diameter(Network_SBM))

fname = "vary_ps"
np.savez(f"{fname}.npz", deltas=deltas, er_ps=er_ps, Thetas=Thetas, Thetas_FwA=Thetas_FwA)

plt.figure(1)
plt.plot(er_ps, deltas)
# plt.legend(["ER", "BA", "SBM"])
plt.title(f"delta vs p")
plt.xlabel("p")
plt.ylabel("delta")
plt.savefig(f"{fname}.pdf", dpi=500)
plt.show()









#
# ## Exp. 1 delta vs. p
# ## create ER networks of different densities
# ER_diameters, ER_components, ER_deltas, ER_regret_gaps = [], [], [], []
# u = 2023
# gamma = 3
# er_ps = np.linspace(0.01, 0.6, 20)
# # er_ps = np.linspace(0.01, 2.0, 20) / N ** (0.5 * (1 - (1 / gamma)))
# n_repeat = 10
# T = int(1e2)
#
# if not os.path.exists("deltas/vary_p/networks"):
#     os.makedirs("deltas/vary_p/networks")
#
#
# def F(Network):
#     return compute_delta(Network, arm_sets_uniform, reward_avgs, K, gamma)
#
#
# def F_regret(Network):
#     Agents = [Agent(v, arm_sets_uniform[v], reward_avgs, Network, 0, 0, K) for v in range(N)]
#
#     regret_gap = 0
#     for _ in range(10):
#         # Flooding
#         problem = create_problem(Network, Agents, T, N, K, ("Flooding", None, gamma, 1.0, n_repeat))
#         Group_Regrets_flooding, _, _, _ = run_ucb(problem, 1.0)
#
#         # FWA
#         problem = create_problem(Network, Agents, T, N, K, ("Flooding-Absorption", None, gamma, 1.0, n_repeat))
#         Group_Regrets_fwa, _, _, _ = run_ucb(problem, 1.0)
#
#         regret_gap += Group_Regrets_fwa[-1] - Group_Regrets_flooding[-1]
#     return regret_gap
#     # for _ in range(n_repeat):
#     #     # Flooding
#     #     problem = create_problem(Network, Agents, T, N, K, (None, "Flooding", gamma, 1.0, n_repeat))
#     #     Group_Regrets_flooding, _, _, _ = run_ucb(problem, 1.0)
#     #
#     #     # FWA
#     #     problem = create_problem(Network, Agents, T, N, K, (None, "Flooding-Absorption", gamma, 1.0, n_repeat))
#     #     Group_Regrets_fwa, _, _, _ = run_ucb(problem, 1.0)
#     #
#     #     regret_gap.append(Group_Regrets_fwa[-1] - Group_Regrets_flooding[-1])
#     # return regret_gap
#
#
# for u in tqdm(range(2023, 2023 + n_repeat)):
#     # theoretical bound (compute delta)
#     Networks, diameters, components = [], [], []
#     for er_p in er_ps:
#         Network = gnx.erdos_renyi_graph(N, er_p, seed=u)
#         Network.name = f"ER_{er_p}"
#         pos = gnx.spring_layout(Network)
#         plot_network(Network, pos, parent="deltas/vary_p/networks")
#         Networks.append(Network)
#
#         ccs = nx.connected_components(Network)
#         components.append(len(list(ccs)))
#         if len(list(ccs)) == 0:
#             diameters.append(0)
#         else:
#             diameters.append(nx.diameter(Network.subgraph(max(ccs, key=len))))
#
#     with Pool() as pool:
#         finals = pool.map_async(F, Networks)
#         finals = finals.get()
#
#     ER_deltas.append(finals)
#     ER_diameters.append(diameters)
#     ER_components.append(components)
#
#     # # empirical regret gap (T=500)
#     # with Pool() as pool:
#     #     finals = pool.map_async(F_regret, ER_Networks)
#     #     finals = finals.get()
#     #
#     # ER_regret_gaps.append(finals)
#
# names = ["delta", "diameter", "component", "regret_gap"]
# for i, ER_i in enumerate([ER_deltas, ER_diameters, ER_components]):
#     # for i, ER_i in enumerate([ER_deltas, ER_regret_gaps]):
#     ER_i = np.array(ER_i)
#     ER_i_means = np.mean(ER_i, axis=0)
#     ER_i_stds = np.std(ER_i, axis=0)
#
#     fname = f"vary_p_{names[i]}"
#
#     np.savez(f"deltas/{fname}.npz", ER_deltas_means=ER_i_means, ER_deltas_stds=ER_i_stds, er_ps=er_ps)
#
#     plt.figure(i)
#     plt.errorbar(er_ps, ER_i_means, ER_i_stds)
#     plt.title(f"ER, {names[i]} vs. p (gamma={gamma})")
#     plt.xlabel("p")
#     plt.ylabel(names[i])
#     plt.savefig(f"deltas/{fname}.pdf", dpi=500)
# plt.show()