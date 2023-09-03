import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import matplotlib.animation

from math import log


num_clusters = 4  # for SBM
size_cluster = 25
N = size_cluster * num_clusters  # number of agents
# er_p = 2 * log(N) / N
er_p = 3 / N  # for large instances
dynamic_p_sparse, dynamic_q_sparse = 1e-3, 9e-3
dynamic_p_dense, dynamic_q_dense = 1e-3, 1e-3

## Erdos-Renyi
# if the graph is disconnected, keep trying other seeds until the graph is connected.
u = 2023
while not nx.is_connected(nx.erdos_renyi_graph(N, er_p, seed=u)):
    u += 1
Network_ER = nx.erdos_renyi_graph(N, er_p, seed=u)
Network_ER.name = f"ER_{er_p}"
pos_ER = nx.spring_layout(Network_ER)

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
    return Network

def update(num, mp):
    ax.clear()
    # update network
    if "dynamic_sparse" in mp:
        update_network(Network, dynamic_p_sparse, dynamic_q_sparse)
    elif "dynamic_dense" in mp:
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


T = int(1e2)
positions = [pos_ER]
for i, original_Network in enumerate([Network_ER]):
    for mp in ["dynamic_sparse"]:
    # for mp in ["dyanamic_sparse", "dyanamic_dense", "dynamic_hybrid"]:
        def update_mp(num):
            update(num, mp)
        Network = copy.deepcopy(original_Network)
        pos = positions[i]

        # Build plot
        fig, ax = plt.subplots()

        ani = matplotlib.animation.FuncAnimation(fig, update_mp, frames=T, interval=10, repeat=False)
        ani.save(f'dynamic_networks/{Network.name}_T_{T}_{mp}.gif', fps=60)
        plt.show()