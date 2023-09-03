from utils import *
import re
from collections import deque
from scipy.stats import cauchy, levy
from math import inf, log
import random

from tqdm import tqdm
import multiprocess
from multiprocess import Pool
from copy import deepcopy
from itertools import combinations


class Problem:
    def __init__(self, Network, Agents, T, N, K, param):
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

    def __str__(self):
        return f"{self.mp}, n_gossip={self.n_gossip}"


class Message:
    def __init__(self, arm, reward, hash_value, v_prev, gamma):
        self.arm = arm
        self.reward = reward
        self.hash_value = hash_value
        self.v_prev = v_prev
        self.gamma = gamma
        self.original_gamma = gamma

    def decay(self):
        self.gamma -= 1

    def update(self, v):
        self.v_prev = v

    def corrupt(self, mp):
        if np.random.binomial(1, 0.5) == 1:
            if "gaussian" in mp:
                self.reward += np.random.normal(0.0, 0.01)
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

        # list of times in which messages get absorbed
        self.message_absorption_times = []

        # for adaptive TTL scheme
        self.returns = {}

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

        if "adaptive" in self.mp:
            # adaptive TTL (if its msg is return too many times, reduce TTL by 1!)
            if len(self.returns) > 0 and max(self.returns.values()) > 3:
                self.gamma = max(self.gamma - 1, 1)

        # poisson clock for message expiration, for testing random stop time.
        if "Poisson" in self.mp:
            new_gamma = np.random.poisson(self.gamma)
        else:
            new_gamma = self.gamma
        # message to be sent to his neighbors
        if new_gamma - 1 >= 0:
            msg_hash = hash((self.idx, arm, reward))
            self.returns[msg_hash] = 0
            return Message(arm, reward, msg_hash, self.idx, new_gamma - 1)
        else:
            return None

    def store_message(self, message):
        message.decay()
        if message.gamma < 0:  # if the message is not yet expired, for MP
            self.message_absorption_times.append(message.original_gamma)
            del message
        else:
            message.update(self.idx)  # update message.v_prev to the current agent
            self.messages.append(message)

    def receive(self, message):
        if message is not None:
            # check for the number of returns of hash values
            if message.hash_value in self.returns.keys():   # if the message is return to the agent
                self.returns[message.hash_value] += 1
            # receive, depending on the communication protocol!
            # RS_p: random stopping probability
            if "RandomStop" in self.mp:
                # if "RandomStop" in problem.mp and np.random.binomial(1, problem.RS_p) == 1:
                RS_p = float(re.findall(r"[\d\.\d]+", self.mp)[0])
                # if it's randomly stopped, then delete the message
                if np.random.binomial(1, RS_p) == 1:
                    # self.message_absorption_times.append(message.original_gamma - message.gamma)
                    del message
                else:
                    if message.arm in self.arm_set:
                        self.total_visitations[message.arm] += 1
                        self.total_rewards[message.arm] += message.reward
                    self.store_message(message)
            else:
                if "Flooding" in self.mp:
                    if message.arm in self.arm_set:
                        self.total_visitations[message.arm] += 1
                        self.total_rewards[message.arm] += message.reward
                        if "Absorption" in self.mp:
                            self.message_absorption_times.append(message.original_gamma - message.gamma)
                            del message
                        else:
                            self.store_message(message)
                    else:
                        self.store_message(message)
                else:
                    del message
        else:
            del message


def update_network(Network, p, q):
    cur_edges = nx.edges(Network)
    cur_non_edges = nx.non_edges(Network)
    for non_edge in cur_non_edges:
        if np.random.binomial(1, p) == 1:
            Network.add_edge(non_edge[0], non_edge[1])
    for edge in cur_edges:
        if np.random.binomial(1, q) == 1:
            Network.remove_edge(edge[0], edge[1])

def run_ucb(problem, p):
    # reseeding
    np.random.seed()

    # initialize parameters
    T, N = problem.T, problem.N
    Agents, Network = problem.Agents, problem.Network
    n_gossip, mp = problem.n_gossip, problem.mp

    # for networks not SBM:
    tmp = list(Network.edges())
    if len(tmp) > 0:
        fixed_edge = tmp[0]
    else:
        fixed_edge = None
    # for logging
    Regrets = [[0 for _ in range(T)] for _ in range(N)]
    Communications = [[0 for _ in range(T)] for _ in range(N)]
    Edge_Messages = [0 for _ in range(T)]

    # run UCB
    # for t in tqdm(range(T)):
    dynamic_p_very_sparse, dynamic_q_very_sparse = 0.0, 0.8
    dynamic_p_sparse, dynamic_q_sparse = 1e-3, 5e-3
    dynamic_p_dense, dynamic_q_dense = 1e-3, 1e-3
    for t in range(T):
        if "dynamic_sparse" in mp:
            update_network(Network, dynamic_p_sparse, dynamic_q_sparse)
        elif "_dynamic_very_sparse" in mp:
            update_network(Network, dynamic_p_very_sparse, dynamic_q_very_sparse)
        elif "dynamic_dense" in mp:
            update_network(Network, dynamic_p_dense, dynamic_q_dense)
        elif "dynamic_hybrid" in mp:
            if t < T // 3:
                update_network(Network, dynamic_p_dense, dynamic_q_dense)
            else:
                update_network(Network, dynamic_p_sparse, dynamic_q_sparse)

        # ## Network switching
        # if Network.name == "Star":
        #     if np.random.binomial(1, 0.5) == 1:
        #         Network = nx.complete_graph(N)  # complete graph
        #     else:  # star graph, with random center!
        #         Network = nx.Graph()
        #         Network.add_nodes_from(range(N))
        #         center = np.random.randint(N)
        #         Network.add_edges_from([(center, i) for i in range(N) if i != center])

        # original_edges = list(Network.edges())
        # # failure of entire packet of messages, for each link
        # linkfailures = {edge: np.random.binomial(1, p) for edge in original_edges}
        # linkfailures.update({(edge[1], edge[0]): linkfailures[edge] for edge in original_edges})

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

            if "baseline" in mp or ("bandwidth" in mp and len(messages) >= 150):
                del messages
            else:
                neighbors = Network.adj[v]
                # message intereference, if applicable
                if "interfere" in mp:
                    messages = interfere(messages)
                # message broadcasting
                while messages:
                    message = messages.pop()
                    # if the message is in-tact (e.g., didn't get destroyed in link failure)
                    if message is not None:
                        # remove the previously originating agent
                        neighbors_new = [nbhd for nbhd in neighbors if nbhd != message.v_prev]
                        # if the message does not hit a dead end, do message passing
                        if len(neighbors_new) > 0:
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
                                # update number of messages for (12, 21) (for small SBM) or (11, 53) (for large SBM)
                                # if (v==12 and neighbor==21) or (v==21 and neighbor==12):
                                if "SBM" in Network.name:
                                    if (v == 11 and neighbor == 53) or (v == 53 and neighbor == 11):
                                        Edge_Messages[t] += 1
                                else:
                                    if fixed_edge is not None:
                                        if (v == fixed_edge[0] and neighbor == fixed_edge[1]) or (v == fixed_edge[1] and neighbor == fixed_edge[0]):
                                            Edge_Messages[t] += 1
                                # delete message if it has already been observed
                                if message_copy.hash_value in Agents[neighbor].history:
                                    del message_copy
                                else:
                                    # send messages
                                    if False:
                                    # if linkfailures[(v, neighbor)] == 0:  # failure of entire packet of messages
                                        Agents[neighbor].receive(None)
                                    else:
                                        # update hash value history
                                        Agents[neighbor].history.append(message_copy.hash_value)
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
    combined_metrics = np.sum(np.log(np.array(Communications) + 2)*Regrets, axis=0)
    # # message absorption times
    # message_absorption_times = []
    # for Agent in Agents:
    #     message_absorption_times += Agent.message_absorption_times

    return Group_Regrets, Group_Communications, np.array(Edge_Messages), combined_metrics


def interfere(messages):
    messages_arms = deque(message.arm for message in messages)
    for message in messages:
        if messages_arms.count(message.arm) > 1:
            # messages.remove(message)
            return deque([None])
    return messages
