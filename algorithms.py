from utils import *
from collections import deque
from scipy.stats import cauchy, levy
from math import inf

from tqdm import tqdm
from copy import deepcopy


class Problem:
    def __init__(self, Network, Agents, T, N, K, param):
        # param: (discard, n_gossip, mp, gamma, p, n_repeat)
        _, _, mp, gamma, _, _ = param

        self.Agents = Agents
        self.Network = Network

        self.T, self.N, self.K = T, N, K

        self.discard = param[0]
        self.n_gossip = param[1]
        self.mp = param[2]
        self.gamma = param[3]
        self.p = param[4]  # *not* link failure (i.e. communication) probability
        self.n_repeat = param[5]

    def __str__(self):
        return f"discard={self.discard}, n_gossip={self.n_gossip}, {self.mp}"


class Message:
    def __init__(self, arm, reward, hash_value, gamma):
        self.arm = arm
        self.reward = reward
        self.hash_value = hash_value
        self.gamma = gamma

    def decay(self):
        self.gamma -= 1

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

    def __init__(self, idx, arm_set, reward_avgs, Network, gamma, mp, q, K):
        self.Network = Network
        self.idx = idx
        self.arm_set = arm_set
        self.optimal = max([reward_avgs[arm] for arm in arm_set])  # local optimal reward
        self.messages = deque()
        self.gamma = gamma
        self.mp = mp
        self.q = q

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
        self.history = set()

    def UCB_network(self):
        # warm-up
        if self.t < len(self.arm_set):
            final_arm = self.arm_set[self.t]
        else:
            final_arm, tmp = None, -inf
            for arm in self.arm_set:
                m = self.total_visitations[arm]
                reward_estimate = self.total_rewards[arm] / m
                bonus = np.sqrt(2 * np.log(self.t) / m)
                ucb = reward_estimate + bonus
                if tmp < ucb:
                    final_arm = arm
                    tmp = ucb
        # finalize the set of messages to be passed along
        cur_message = self.pull(final_arm)
        self.history.add(cur_message.hash_value)
        self.messages.append(cur_message)

        # empty his current set of messages
        final_messages = self.messages
        if "interfere" in self.mp:
            final_messages = interfere(final_messages)
        self.messages = deque()
        return final_messages

    def pull(self, arm):
        if arm not in self.arm_set:
            raise ValueError(f"{arm} not in the arm set {self.arm_set} attained by agent {self.idx}")

        # update number of visitations
        self.total_visitations[arm] += 1

        # receive/update reward
        reward = np.random.binomial(1, self.reward_avgs[arm])
        self.total_rewards[arm] += reward

        # update regret
        self.regret += self.optimal - self.reward_avgs[arm]

        # next time step
        self.t += 1

        # message to be sent to his neighbors
        return Message(arm, reward, hash((arm, reward)), self.gamma)

    def store_message(self, message):
        message.decay()
        if "corrupt" in self.mp:
            message.corrupt(self.mp)
        if message.gamma < 0:  # if the message is not yet expired, for MP
            del message
        else:
            self.messages.append(message)

    def receive_message(self, message):
        # delete message if it has already been observed
        if message.hash_value in self.history:
            del message
        else:
            arm, reward = message.arm, message.reward
            self.history.add(message.hash_value)

            self.total_visitations[arm] += 1
            self.total_rewards[arm] += reward

    def receive(self, messages, p_v=1):
        while messages:  # if d is True if d is not empty (canonical way for all collections)
            message = messages.pop()
            if message is not None:
                contain_message = message.arm in self.arm_set
                if np.random.binomial(1, p_v) == 1:  # if discarding does not take place
                    # receive, depending on the communication protocol!
                    if "MP" in self.mp:
                        if "Hitting" in self.mp:
                            if contain_message:
                                self.receive_message(message)
                                del message
                            else:
                                self.store_message(message)
                        else:
                            if contain_message:
                                self.receive_message(message)
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
    discard, n_gossip, mp = problem.discard, problem.n_gossip, problem.mp

    # for logging
    Regrets = [[0 for _ in range(T)] for _ in range(N)]
    Communications = [[0 for _ in range(T)] for _ in range(N)]

    min_deg = min((d for _, d in Network.degree()))
    # run UCB
    # for t in tqdm(range(T)):
    original_edges = Network.edges()
    for t in range(T):
        # # fail edges randomly w.p. 1-p, i.i.d. -> for temporally changing graphs
        # failed_edges = [edge for edge in original_edges if np.random.binomial(1, p) == 0]
        # Network_modified = deepcopy(Network)
        # Network_modified.remove_edges_from(failed_edges)

        # single iteration of UCB
        for i in range(N):
            messages = Agents[i].UCB_network()
            neighbors = Network.adj[i]

            if "bandwidth" in mp and len(messages) >= 5:
                neighbors = []

            while messages:
                message = messages.pop()
                # construct neighbors to which the message will be gossiped to (push protocol!)
                if n_gossip is None or n_gossip >= len(neighbors):
                    gossip_neighbors = neighbors
                else:
                    gossip_neighbors = np.random.choice(neighbors, size=n_gossip, replace=False)
                # pass messages
                for neighbor in gossip_neighbors:
                    message_copy = deepcopy(message)
                    # update communication complexity
                    Agents[i].communication += 1
                    # send messages
                    if np.random.binomial(1, p) == 0:  # message failure
                        Agents[neighbor].receive(deque([None]), Agents[i])
                    else:
                        if discard:
                            Agents[neighbor].receive(deque(message_copy), min_deg / Network.degree[neighbor])
                        else:
                            Agents[neighbor].receive(deque([message_copy]))
                # delete the message sent by the originating agent, after communication is complete
                del message

        # collect regrets and communications
        for i in range(N):
            Regrets[i][t], Communications[i][t] = Agents[i].regret, Agents[i].communication

    # group regrets and communications
    Group_Regrets = np.sum(np.array(Regrets), axis=0)
    Group_Communications = np.sum(np.array(Communications), axis=0)

    # # plotting quantities vs. iteration
    # path = f"heterogeneous/{mp}_n_gossip={n_gossip}_discard={discard}"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #
    # # plot regrets
    # titles = [f"{mp}: {wise} Regrets ({Network.name}, p={p}, gamma={Agents[0].gamma}, n_gossip={n_gossip}, " \
    #           f"Discard={discard})" for wise in ["Agent-specific", "Group"]]
    # fnames = [f"{path}/Regret_{wise}_{mp}_p={p}_gamma={Agents[0].gamma}_{Network.name}.pdf" for wise in
    #           ["agent", "group"]]
    # plot(Regrets, Group_Regrets, Network, titles, fnames)
    #
    # # plot communications
    # titles = [f"{mp}: {wise} Communications ({Network.name}, p={p}, gamma={Agents[0].gamma}, n_gossip={n_gossip}, " \
    #           f"Discard={discard})" for wise in ["Agent-specific", "Group"]]
    # fnames = [f"{path}/Communication_{wise}_{mp}_p={p}_gamma={Agents[0].gamma}_{Network.name}.pdf" for wise in
    #           ["agent", "group"]]
    # plot(Communications, Group_Communications, Network, titles, fnames)

    return Group_Regrets[-1], Group_Communications[-1]


def interfere(messages):
    # if len(tmp) > 5:
    #     self.messages = deque(np.random.choice(self.messages, 5))
    # return self.messages
    messages_arms = deque(message.arm for message in messages)
    for message in messages:
        if messages_arms.count(message.arm) > 1:
            # messages.remove(message)
            return deque([None])
    return messages
