from math import inf
from copy import deepcopy
from itertools import combinations
import grinpy as gnx
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


def compute_invariants(Network, arm_sets, reward_avgs, K, gamma):
    # tilde{Delta]_a^v as in Yang et al., INFOCOM 2022
    Deltas = []
    for a in range(K):
        Delta = inf
        for v in Network.nodes:
            arm_set = arm_sets[v]
            max_reward = max([reward_avgs[a] for a in arm_set])
            if a in arm_set:
                delta = max_reward - reward_avgs[a]
                if Delta > delta > 0:
                    Delta = delta
        if Delta == inf:
            Delta = 0
        Deltas.append(Delta)

    # Compute theta([G^gamma]_{-a})
    Thetas, Thetas_FWA = [], []
    Network_gamma = gnx.power(Network, gamma)
    output = 0
    for a in range(K):
        Agents_a = [v for v in Network.nodes if
                    a in arm_sets[v] and max([reward_avgs[a] for a in arm_sets[v]]) - reward_avgs[a] > 0]

        Network_gamma_a = Network_gamma.subgraph(Agents_a)
        theta = gnx.chromatic_number(gnx.complement(Network_gamma_a))  # theta(G) = chromatic_number(complement of G)
        Thetas.append(theta)

        Network_gamma_a_nonblocking = non_blocking_power(Network, arm_sets, gamma, a)
        Network_gamma_a_nonblocking = Network_gamma_a_nonblocking.subgraph(Agents_a)
        theta_FWA = gnx.chromatic_number(gnx.complement(Network_gamma_a_nonblocking))
        Thetas_FWA.append(theta_FWA)

        output += int(Deltas[a] > 0) * (theta_FWA - theta)

    return list(zip(Deltas, Thetas, Thetas_FWA)), output


def compute_delta(Network, arm_sets, reward_avgs, K, gamma):
    tmp, delta = compute_invariants(Network, arm_sets, reward_avgs, K, gamma)
    return delta


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
