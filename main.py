from utilities import *
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_ucb(Agents, Network):
    # initialize agent-specific regrets (for logging)
    Regrets = [[0 for _ in range(T)] for _ in range(N)]

    # run UCB
    for t in tqdm(range(1, T + 1)):
        # single iteration of UCB
        for i in range(N):
            message = Agents[i].UCB_network()
            for nbh in Network.adj[i]:
                # link failure
                if np.random.binomial(1, p) == 0:
                    Agents[nbh].receive(None)
                else:
                    Agents[nbh].receive(message)

        # collect regrets
        for i in range(N):
            Regrets[i][t] = Agents[i].regret
    return Regrets


def plot(Regrets):
    # individual regret
    plt.figure(1)
    for i in range(N):
        plt.plot(Regrets[i])
    plt.title("Agent-specific regrets (UCB-Network)")
    plt.legend([f"Agent {i} (degree = {network.degree(i)})" for i in range(N)])
    plt.savefig("agent-regret-ucb-network.pdf", dpi=1200)
    plt.show()

    # group regret
    Group_Regrets = np.sum(np.array(Regrets), axis=0)
    plt.figure(2)
    plt.plot(Group_Regrets)
    plt.title("Group Regret")
    plt.savefig("group-regret-ucb-network.pdf", dpi=1200)
    plt.show()


if __name__ == '__main__':
    N, K = 20, 10  # number of agents, total number of arms
    T = int(1e6)  # number of iterations
    p = 0.9  # probability that a message is *not* discarded!
    RG_model = "ER"

    # arm set and their mean rewards
    total_arm_set = list(range(K))
    """
    reward: dict of reward to its mean
    """
    tmp = np.linspace(0.1, 1.0, num=K)
    reward_avgs = {total_arm_set[i]: tmp[i] for i in range(K)}

    # create communication network
    if RG_model == "ER":
        Network = nx.erdos_renyi_graph(N, 0.2)
    elif RG_model == "BA":
        Network = nx.barabasi_albert_graph(N, 20)
    else:
        raise NotImplementedError(f"{RG_model} not yet implemented")

    # create agents
    Agents = []
    for i in range(N):
        arm_set_i = total_arm_set   # Change to heterogeneous case!
        Agents.append(Agent(i, arm_set_i, reward_avgs))

    Regrets = run_ucb(Agents, Network)

    # plot
    plot(Regrets)
