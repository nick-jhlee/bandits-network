from math import inf
from copy import deepcopy
from itertools import combinations
import grinpy as gnx
import networkx as nx
import numpy as np


def non_blocking_power(Network, arm_sets, gamma, a):
    Network_result = deepcopy(Network)
    for v, w in combinations(Network.nodes, 2):
        if gnx.has_path(Network, v, w) or gnx.has_path(Network, w, v):
            try:
                for path in gnx.all_simple_paths(Network, v, w, gamma):
                    if len(path) == 1:
                        raise ValueError
                    else:
                        tmp = True
                        for u in path[1:-1]:
                            if a in arm_sets[u]:
                                tmp = False
                        if tmp:
                            raise ValueError
            except:
                Network_result.add_edge(v, w)
    return Network_result


def compute_delta(Network, arm_sets, K, gamma, Deltas, Deltas_tilde, mode="greedy"):
    # Compute theta([G^gamma]_{-a})
    Thetas, Thetas_FWA = [], []
    Network_gamma = gnx.power(Network, gamma)
    output = 0
    for a in range(K):
        if Deltas_tilde[a] > 0:
            Agents_a = [v for v in Network.nodes if a in arm_sets[v] and Deltas[a][v] > 0]

            Network_gamma_a = Network_gamma.subgraph(Agents_a)
            if mode == "greedy":
                coloring = nx.coloring.greedy_color(nx.complement(Network_gamma_a), strategy="largest_first")
                theta = len(list(set(coloring.values())))
            else:
                theta = gnx.chromatic_number(
                    gnx.complement(Network_gamma_a))  # theta(G) = chromatic_number(complement of G)
            Thetas.append(theta)

            Network_gamma_a_nonblocking = non_blocking_power(Network, arm_sets, gamma, a)
            Network_gamma_a_nonblocking = Network_gamma_a_nonblocking.subgraph(Agents_a)
            if mode == "greedy":
                coloring = nx.coloring.greedy_color(nx.complement(Network_gamma_a_nonblocking), strategy="largest_first")
                theta_FWA = len(list(set(coloring.values())))
            else:
                theta_FWA = gnx.chromatic_number(gnx.complement(Network_gamma_a_nonblocking))
            Thetas_FWA.append(theta_FWA)

            output += (theta_FWA - theta) / Deltas_tilde[a]

    # print(Deltas)
    # print(Deltas_tilde)
    # print(Thetas)
    # print(Thetas_FWA)
    return Thetas, Thetas_FWA, output


def uniform_arm_sets(N, K):
    arm_sets = []  # uniformly random
    for agent in range(N):
        arm_set = []
        while len(arm_set) == 0:
            for arm in range(K):
                if np.random.rand() < 0.5:
                    arm_set.append(arm)
        arm_sets.append(arm_set)
    return arm_sets
