import numpy as np


class Agent():
    """
    reward_avgs: dict of reward to its mean
    """

    def __init__(self, idx, arm_set, reward_avgs):
        self.idx = idx
        self.arm_set = arm_set
        self.optimal = max([reward_avgs[arm] for arm in arm_set])

        self.reward_avgs = reward_avgs

        # iteration
        self.t = 0

        # his own regret
        self.regret = 0

        # including neighbors
        self.total_rewards = dict.fromkeys(arm_set, 0)
        self.total_visitations = dict.fromkeys(arm_set, 0)

    def UCB_network(self):
        if self.t <= len(self.arm_set):
            return self.arm_set[self.t]
        else:
            final_arm, tmp = None, 0
            for arm in arm_set:
                m = self.total_visitations[arm]
                reward_estimate = self.total_rewards[arm] / m
                bonus = np.sqrt(2 * np.log(self.t) / m)
                ucb = reward_estimate + bonus
                if tmp < ucb:
                    final_arm = arm
                    tmp = ucb
            message = self.pull(final_arm)
            print("message: ", message)
            return message

    def pull(self, arm):
        # update number of visitations
        self.total_visitations[arm] += 1

        # receive reward
        if arm not in arm_set:
            raise ValueError(f"{arm} not in the arm set attained by agent {self.idx}")

        # update reward
        reward = np.random.binomial(1, self.reward_avgs[arm])
        self.total_rewards[arm] += reward

        # update regret
        self.regret += self.optimal - self.reward_avgs[arm]

        # next time step
        self.t += 1

        # message to be sent to his neighbors
        return arm, reward

    def receive(self, message):
        if message is not None:
            arm, reward = message
            self.total_visitations[arm] += 1
            self.total_rewards[arm] += reward
