import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import matplotlib.animation

from math import log


num_clusters = 2  # for SBM
size_cluster = 20
# size_cluster = 10
N = size_cluster * num_clusters  # number of agents
# er_p = 2 * log(N) / N
er_p = 1.001 * log(N) / N  # for large instances
dynamic_p_sparse, dynamic_q_sparse = 0.04 / N, 1 / N
dynamic_p_dense, dynamic_q_dense = 0.1 / N, 0.2 / N

## Erodos-Renyi
# if the graph is disconnected, keep trying other seeds until the graph is connected.
u = 2023
while not nx.is_connected(nx.erdos_renyi_graph(N, er_p, seed=u)):
    u += 1
Network_ER = nx.erdos_renyi_graph(N, er_p, seed=u)
Network_ER.name = f"ER_{er_p}"
pos_ER = nx.spring_layout(Network_ER)

# Barabasi-Albert
# m = 5  # for small instances
m = 3
Network_BA = nx.barabasi_albert_graph(N, m, seed=2023)
Network_BA.name = f"BA_{m}"
pos_BA = nx.spring_layout(Network_BA)

## Binary SBM
# sbm_p, sbm_q = 2 * er_p, 0.01
sbm_p, sbm_q = 2 * er_p, 0.001  # for large instances
u = 2023
while not nx.is_connected(
        nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=u)):
    u += 1
Network_SBM = nx.random_partition_graph([size_cluster for _ in range(num_clusters)], sbm_p, sbm_q, seed=u)
Network_SBM.name = f"SBM_{sbm_p}_{sbm_q}"
pos_SBM = nx.spring_layout(Network_SBM)

if not os.path.exists(f"dynamic_networks"):
    os.makedirs(f"dynamic_networks")


def update_network(Network, p, q):
    cur_edges = nx.edges(Network)
    cur_non_edges = nx.non_edges(Network)
    for non_edge in cur_non_edges:
        if np.random.binomial(1, p) == 1:
            Network.add_edge(non_edge[0], non_edge[1])
    for edge in cur_edges:
        if np.random.binomial(1, q) == 1:
            Network.remove_edge(edge[0], edge[1])


def update(num, mp):
    ax.clear()
    # update network
    if "dyanamic_sparse" in mp:
        update_network(Network, dynamic_p_sparse, dynamic_q_sparse)
    elif "dyanamic_dense" in mp:
        update_network(Network, dynamic_p_dense, dynamic_q_dense)
    elif "dynamic_hybrid" in mp:
        if num < 30:
            update_network(Network, dynamic_p_dense, dynamic_q_dense)
        else:
            update_network(Network, dynamic_p_sparse, dynamic_q_sparse)

    nx.draw_networkx(Network, with_labels=True, pos=pos, node_size=100, font_size=8)

    # Scale plot ax
    ax.set_xticks([])
    ax.set_yticks([])


positions = [pos_ER, pos_BA, pos_SBM]
for i, original_Network in enumerate([Network_ER]):
    for mp in ["dyanamic_sparse", "dyanamic_dense", "dynamic_hybrid"]:
        def update_mp(num):
            update(num, mp)
        Network = copy.deepcopy(original_Network)
        pos = positions[i]

        # Build plot
        fig, ax = plt.subplots()

        ani = matplotlib.animation.FuncAnimation(fig, update_mp, frames=100, interval=1, repeat=False)
        ani.save(f'dynamic_networks/{Network.name}_T_100_{mp}.gif', fps=60)
        plt.show()