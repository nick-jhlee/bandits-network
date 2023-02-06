from utils import *
from tqdm import tqdm

import os


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
        self.p = param[4]
        self.n_repeat = param[5]

    def __str__(self):
        return f"discard={self.discard}, n_gossip={self.n_gossip}, {self.mp}"


class Message:
    def __init__(self, arm, reward, origin, gamma):
        self.arm = arm
        self.reward = reward
        self.origin = origin
        self.gamma = gamma

    def decay(self):
        self.gamma -= 1


class Agent:
    """
    reward_avgs: dict of reward to its mean
    """

    def __init__(self, idx, arm_set, reward_avgs, Network, gamma, mp):
        self.Network = Network
        self.idx = idx
        self.arm_set = arm_set
        self.optimal = max([reward_avgs[arm] for arm in arm_set])
        self.messages = []
        self.gamma = gamma
        self.mp = mp

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

    def UCB_network(self):
        # warm-up
        if self.t < len(self.arm_set):
            final_arm = self.arm_set[self.t]
        else:
            final_arm, tmp = None, 0
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
        final_messages = self.messages + [cur_message]

        # empty his current set of messages
        self.messages = []
        return final_messages

    def pull(self, arm):
        # update number of visitations
        self.total_visitations[arm] += 1

        # receive/update reward
        if arm not in self.arm_set:
            raise ValueError(f"{arm} not in the arm set attained by agent {self.idx}")
        reward = np.random.binomial(1, self.reward_avgs[arm])
        self.total_rewards[arm] += reward

        # update regret
        self.regret += self.optimal - self.reward_avgs[arm]

        # next time step
        self.t += 1

        # message to be sent to his neighbors
        return Message(arm, reward, self.idx, self.gamma)

    def store_message(self, message):
        message.decay()
        self.messages.append(message)

    def receive_message(self, message):
        arm, reward = message.arm, message.reward
        self.total_visitations[arm] += 1
        self.total_rewards[arm] += reward

    def receive(self, messages, origin, p_v=1):
        for message in messages:
            if message is not None:
                contain_message = message.arm in self.arm_set
                if message.gamma > 0:  # if the message is not yet expired
                    message.origin = origin  # update origin
                    if np.random.binomial(1, p_v) == 1:  # if discarding does not take place
                        if self.mp == "MP" or (self.mp == "Greedy-MP" and contain_message):
                            if contain_message:
                                self.receive_message(message)
                            self.store_message(message)
                        elif self.mp == "Hitting-MP":
                            if contain_message:
                                self.receive_message(message)
                                del message
                            else:
                                self.store_message(message)
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
    for t in tqdm(range(T)):
        # single iteration of UCB
        for i in range(N):
            messages = Agents[i].UCB_network()
            neighbors = Network.adj[i]
            for message in messages:
                # construct gossiping neighborhood
                effective_nbhd = [nhb for nhb in neighbors if nhb != message.origin]
                if len(effective_nbhd) > 0:  # as long as the current message did not hit a dead end
                    if n_gossip is None or n_gossip >= len(effective_nbhd):
                        gossip_neighbors = effective_nbhd
                    else:
                        gossip_neighbors = np.random.choice(effective_nbhd, size=n_gossip, replace=False)
                    # pass messages
                    for neighbor in gossip_neighbors:
                        # update communication complexity
                        Agents[i].communication += 1
                        # send messages
                        if np.random.binomial(1, p) == 0:  # link failure
                            Agents[neighbor].receive([None], Agents[i])
                        else:
                            if discard:
                                Agents[neighbor].receive([message], Agents[i], min_deg / Network.degree[neighbor])
                            else:
                                Agents[neighbor].receive([message], Agents[i])

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
