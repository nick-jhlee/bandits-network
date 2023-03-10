from utils import *
from collections import deque
from scipy.stats import cauchy, levy
from math import inf

from tqdm import tqdm
import multiprocess
from multiprocess import Pool
from copy import deepcopy
from itertools import combinations


class Problem:
    def __init__(self, Network, Agents, T, N, K, param, naive_p):
        # param: (mp, n_gossip, gamma, p, n_repeat)
        mp, n_gossip, gamma, p, n_repeat = param

        self.Agents = Agents
        self.Network = Network

        self.T, self.N, self.K = T, N, K

        self.n_gossip = n_gossip
        self.mp = mp
        self.gamma = gamma
        self.p = p  # communication probability (1 - link failure)
        self.n_repeat = n_repeat

        self.naive_p = naive_p

    def __str__(self):
        return f"{self.mp}, n_gossip={self.n_gossip}"


class Message:
    def __init__(self, arm, reward, hash_value, v_prev, gamma):
        self.arm = arm
        self.reward = reward
        self.hash_value = hash_value
        self.v_prev = v_prev
        self.gamma = gamma

    def decay(self):
        self.gamma -= 1

    def update(self, v):
        self.v_prev = v

    def corrupt(self, mp):
        if np.random.binomial(1, 0.5) == 1:
            if "gaussian" in mp:
                self.reward += np.random.normal(0.1, 0.01)
            elif "zero" in mp:
                self.reward = 0
            elif "heavy" in mp:
                self.reward += levy.rvs()


# class Arm:
#     def __init__(self, arm_idx, reward):
#         self.arm_idx = arm_idx
#         self.reward = reward
#
#     def rot(self, rho=1.0):
#         self.reward *= rho


class Agent:
    """
    reward_avgs: dict of reward to its mean
    """

    def __init__(self, idx, arm_set, reward_avgs, Network, gamma, mp, K):
        self.Network = Network
        self.idx = idx
        self.arm_set = arm_set
        self.optimal = max([reward_avgs[arm] for arm in arm_set])  # local optimal reward
        self.messages = deque()
        self.gamma = gamma
        self.mp = mp

        # not available to the agents; this is for implementeing the arm corruption by the environment
        self.total_arm_set = deque(range(K))

        self.reward_avgs = reward_avgs

        # iteration
        self.t = 0

        # his own regret
        self.regret = 0

        # his own communication complexity (# of messages he passed along)
        self.communication = 0

        # including neighbors
        self.total_rewards = dict.fromkeys(arm_set, 0)
        self.total_visitations = dict.fromkeys(arm_set, 0)

        # hash values of messages that he has seen
        self.history = deque()

    # one round of UCB_network
    def UCB_network(self):
        # warm-up
        if self.t < len(self.arm_set):
            final_arm = self.arm_set[self.t]
        else:
            final_arm, tmp = None, -inf
            for arm in self.arm_set:
                m = self.total_visitations[arm]
                reward_estimate = self.total_rewards[arm] / m
                bonus = np.sqrt(3 * np.log(self.t) / m)
                ucb = reward_estimate + bonus
                if tmp < ucb:
                    final_arm = arm
                    tmp = ucb
        # finalize the set of messages to be passed along
        cur_message = self.pull(final_arm)
        self.messages.append(cur_message)

        # empty his current set of messages
        final_messages = self.messages
        self.messages = deque()
        return final_messages

    def pull(self, arm):
        if arm not in self.arm_set:
            raise ValueError(f"{arm} not in the arm set {self.arm_set} attained by agent {self.idx}")

        # update number of visitations
        self.total_visitations[arm] += 1

        # receive/update reward
        # reward = np.random.binomial(1, self.reward_avgs[arm]) # Berounlli reward
        reward = np.random.normal(self.reward_avgs[arm], 1)
        self.total_rewards[arm] += reward

        # update regret
        self.regret += self.optimal - self.reward_avgs[arm]

        # next time step
        self.t += 1

        # message to be sent to his neighbors
        return Message(arm, reward, hash((self.idx, arm, reward)), self.idx, self.gamma - 1)

    def store_message(self, message):
        message.decay()
        if message.gamma < 0:  # if the message is not yet expired, for MP
            del message
        else:
            message.update(self.idx)  # update message.v_prev to the current agent
            self.messages.append(message)

    def receive(self, message):
        if message is not None:
            # receive, depending on the communication protocol!
            if "Flooding" in self.mp:
                if message.arm in self.arm_set:
                    self.total_visitations[message.arm] += 1
                    self.total_rewards[message.arm] += message.reward
                    if "Absorption" in self.mp:
                        del message
                    else:
                        self.store_message(message)
                else:
                    self.store_message(message)
            else:
                del message
        else:
            del message


def run_ucb(problem, p):
    # reseeding
    np.random.seed()

    # initialize parameters
    T, N = problem.T, problem.N
    Agents, Network = problem.Agents, problem.Network
    n_gossip, mp = problem.n_gossip, problem.mp
    original_edges = list(Network.edges())

    # for logging
    Regrets = [[0 for _ in range(T)] for _ in range(N)]
    Communications = [[0 for _ in range(T)] for _ in range(N)]
    Edge_Messages = [[0 for _ in range(T)] for _ in range(len(original_edges))]

    # run UCB
    # for t in tqdm(range(T)):
    for t in range(T):
        # # fail edges randomly w.p. 1-p, i.i.d. -> for temporally changing graphs
        # failed_edges = [edge for edge in original_edges if np.random.binomial(1, p) == 0]
        # Network_modified = deepcopy(Network)
        # Network_modified.remove_edges_from(failed_edges)

        total_messages = [[None] for _ in range(N)]

        # single iteration of UCB, for each agent!
        for v in range(N):
            total_messages[v] = Agents[v].UCB_network()
            Regrets[v][t] = Agents[v].regret

        ## TODO: implement parallel loop here using ThreadPool!
        ## cf. https://superfastpython.com/parallel-nested-for-loops-in-python
        # def U(v):
        #     messages_v = Agents[v].UCB_network()
        #     regrets_v = Agents[v].regret
        #     return messages_v, regrets_v
        #
        # # run the experiments in parallel
        # with Pool() as pool:
        #     everything = pool.map_async(U, range(N))
        #     everything = everything.get()
        #
        # for v, (messages_v, regrets_v) in enumerate(everything):
        #     total_messages[v] = messages_v
        #     for t, regret in enumerate(regrets_v):
        #         Regrets[v][t] = regret

        # information sharing(dissemination)
        for v in range(N):
            messages = total_messages[v]

            if mp == "baseline" or ("bandwidth" in mp and len(messages) >= 150):
                del messages
            else:
                neighbors = Network.adj[v]
                # message intereference, if applicable
                if "interfere" in problem.mp:
                    messages = interfere(messages)
                # message broadcasting
                # naive_p: random stopping probability
                if "naive" in problem.mp and np.random.binomial(1, problem.naive_p) == 1:
                    del messages
                else:
                    while messages:
                        message = messages.pop()
                        if message is None:
                            del message
                        else:
                            # remove the previously originating agent
                            neighbors_new = [nbhd for nbhd in neighbors if nbhd != message.v_prev]
                            # if the message hits a dead end, delete it
                            if len(neighbors_new) == 0:
                                del message
                            else:
                                # construct neighbors to which the message will be gossiped to (push protocol)
                                if n_gossip is None or n_gossip >= len(neighbors_new):
                                    gossip_neighbors = neighbors_new
                                else:
                                    gossip_neighbors = np.random.choice(neighbors_new, size=n_gossip, replace=False)
                                # pass messages
                                for neighbor in gossip_neighbors:
                                    message_copy = deepcopy(message)
                                    # update communication complexity
                                    Agents[v].communication += 1
                                    # delete message if it has already been observed
                                    if message_copy.hash_value in Agents[neighbor].history:
                                        del message_copy
                                    else:
                                        # update hash value history
                                        Agents[neighbor].history.append(message_copy.hash_value)
                                        # update number of messages per edge
                                        if v < neighbor:
                                            Edge_Messages[original_edges.index((v, neighbor))][t] += 1
                                        else:
                                            Edge_Messages[original_edges.index((neighbor, v))][t] += 1
                                        # send messages
                                        if np.random.binomial(1, p) == 0:  # message failure
                                            Agents[neighbor].receive(None)
                                        else:
                                            if "corrupt" in problem.mp:
                                                message_copy.corrupt(problem.mp)
                                            Agents[neighbor].receive(message_copy)
                        # delete the message sent by the originating agent, after communication is complete
                        del message

            Communications[v][t] = Agents[v].communication

    # group regrets and communications
    # Group_Regrets = np.max(np.array(Regrets), axis=0)
    Group_Regrets = np.sum(np.array(Regrets), axis=0)
    Group_Communications = np.sum(np.array(Communications), axis=0)

    return Group_Regrets, Group_Communications, np.array(Edge_Messages)


def interfere(messages):
    messages_arms = deque(message.arm for message in messages)
    for message in messages:
        if messages_arms.count(message.arm) > 1:
            # messages.remove(message)
            return deque([None])
    return messages


def non_blocking_power(Network, arm_sets, gamma, a):
    Network_result = deepcopy(Network)
    for v, w in combinations(Network.nodes, 2):
        if nx.has_path(Network, v, w) or nx.has_path(Network, w, v):
            try:
                for path in nx.all_simple_paths(Network, v, w, gamma):
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
            if a in arm_set:
                delta = max(reward_avgs[arm_set]) - reward_avgs[a]
                if Delta > delta > 0:
                    Delta = delta
        if Delta == inf:
            Delta = 0
        Deltas.append(Delta)

    # Compute theta([G^gamma]_{-a})
    Thetas, Thetas_FWA = [], []
    Network_gamma = nx.power(Network, gamma)
    output = 0
    for a in range(K):
        Agents_a = [v for v in Network.nodes if a in arm_sets[v] and max(reward_avgs[arm_sets[v]]) - reward_avgs[a] > 0]

        Network_gamma_a = Network_gamma.subgraph(Agents_a)
        theta = nx.chromatic_number(nx.complement(Network_gamma_a))  # theta(G) = chromatic_number(complement of G)
        Thetas.append(theta)

        Network_gamma_a_nonblocking = non_blocking_power(Network, arm_sets, gamma, a)
        Network_gamma_a_nonblocking = Network_gamma_a_nonblocking.subgraph(Agents_a)
        theta_FWA = nx.chromatic_number(nx.complement(Network_gamma_a_nonblocking))
        Thetas_FWA.append(theta_FWA)

        output += int(Deltas[a] > 0) * (theta_FWA - theta)

    return list(zip(Deltas, Thetas, Thetas_FWA)), output
