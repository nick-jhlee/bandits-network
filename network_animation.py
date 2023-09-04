import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import matplotlib.animation
from copy import deepcopy


num_clusters = 4  # for SBM
size_cluster = 25
N = size_cluster * num_clusters  # number of agents
# er_p = 2 * log(N) / N
er_p = 3 / N  # for large instances

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
    Network_copy = deepcopy(Network)
    cur_edges = nx.edges(Network_copy)
    cur_non_edges = nx.non_edges(Network_copy)
    for non_edge in cur_non_edges:
        if np.random.binomial(1, p) == 1:
            Network_copy.add_edge(non_edge[0], non_edge[1])
    for edge in cur_edges:
        if np.random.binomial(1, q) == 1:
            Network_copy.remove_edge(edge[0], edge[1])
    return Network_copy

T = int(1e2)
dynamic_p_sparse, dynamic_q_sparse = 1e-2, 3e-1
dynamic_p_dense, dynamic_q_dense = 1e-2, 1e-1

pos = pos_ER
for mp in ["dynamic_sparse", "dynamic_dense"]:
    # Build plot
    fig, ax = plt.subplots()

    Network = Network_ER
    nx.draw_networkx(Network, with_labels=True, pos=pos, node_size=100, font_size=8)

    def update(num):
        global Network
        ax.clear()
        # update network
        if "dynamic_sparse" in mp:
            Network = update_network(Network, dynamic_p_sparse, dynamic_q_sparse)
        elif "dynamic_dense" in mp:
            Network = update_network(Network, dynamic_p_dense, dynamic_q_dense)
        elif "dynamic_hybrid" in mp:
            if num < 30:
                Network = update_network(Network, dynamic_p_dense, dynamic_q_dense)
            else:
                Network = update_network(Network, dynamic_p_sparse, dynamic_q_sparse)

        nx.draw_networkx(Network, with_labels=True, pos=pos, node_size=100, font_size=8)

        # Scale plot ax
        ax.set_xticks([])
        ax.set_yticks([])

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=T, interval=10, repeat=False)
    ani.save(f'dynamic_networks/{Network.name}_T_{T}_{mp}.gif', fps=60)
    # plt.show()

    T = int(1e3)
    num_edges_sparse, num_edges_dense = [], []

    for _ in range(10):
        Network = Network_ER
        tmp = []
        for t in range(T):
            Network = update_network(Network, dynamic_p_sparse, dynamic_q_sparse)
            tmp.append(nx.number_of_edges(Network))
        num_edges_sparse.append(tmp)

    for _ in range(10):
        Network = Network_ER
        tmp = []
        for t in range(T):
            Network = update_network(Network, dynamic_p_dense, dynamic_q_dense)
            tmp.append(nx.number_of_edges(Network))
        num_edges_dense.append(tmp)

    num_edges_sparse = np.array(num_edges_sparse)
    num_edges_dense = np.array(num_edges_dense)
    #error plot
    plt.plot(range(1,T+1), np.mean(num_edges_dense, axis=0), label="dense")
    plt.fill_between(range(1,T+1), np.mean(num_edges_dense, axis=0) - np.std(num_edges_dense, axis=0), np.mean(num_edges_dense, axis=0) + np.std(num_edges_dense, axis=0),
                                alpha=0.3)
    plt.plot(range(1,T+1), np.mean(num_edges_sparse, axis=0), label="sparse")
    plt.fill_between(range(1,T+1), np.mean(num_edges_sparse, axis=0) - np.std(num_edges_sparse, axis=0), np.mean(num_edges_sparse, axis=0) + np.std(num_edges_sparse, axis=0),
                                alpha=0.3)
    plt.savefig(f'dynamic_networks/num_edges.pdf', dpi=1200, bbox_inches='tight')
    # plt.show()