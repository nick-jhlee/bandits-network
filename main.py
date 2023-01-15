from utilities import *
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocess as mp


# from multiprocessing import Pool


def run_ucb(Agents, Network, p, RG_model):
    # initialize agent-specific regrets (for logging)
    Regrets = [[0 for _ in range(T)] for _ in range(N)]

    # run UCB
    for t in tqdm(range(T)):
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

    plot(Regrets, Network, N, p, RG_model)
    # return Regrets


def plot(Regrets, Network, N, p, RG_model):
    # individual regret
    plt.figure(1)
    order = np.argsort([Network.degree(i) for i in range(N)])
    for i in range(N):
        plt.plot(Regrets[order[i]])
    plt.title(f"Agent-specific regrets (UCB-Network), p={p}, {RG_model}")
    # sort the legend by degree
    plt.legend([f"Agent {order[i]} (deg = {Network.degree(order[i])})" for i in range(N)],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"ucb-network/agent_p={p}_{RG_model}.pdf", dpi=1200, bbox_inches='tight')
    plt.show()

    # group regret
    Group_Regrets = np.sum(np.array(Regrets), axis=0)
    plt.figure(2)
    plt.plot(Group_Regrets)
    plt.title("Group Regret")
    plt.savefig(f"ucb-network/group_p={p}_{RG_model}.pdf", dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    N, K = 40, 20  # number of agents, total number of arms
    T = int(1e6)  # number of iterations
    ps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # probability that a message is *not* discarded!

    # arm set and their mean rewards
    total_arm_set = list(range(K))
    tmp = np.linspace(0.1, 1.0, num=K)
    reward_avgs = {total_arm_set[i]: tmp[i] for i in range(K)}

    # create agents
    Agents = []
    for i in range(N):
        arm_set_i = total_arm_set  # Change to heterogeneous case!
        Agents.append(Agent(i, arm_set_i, reward_avgs))

    for RG_model in ["BA", "ER"]:
        # create communication network
        if RG_model == "ER":
            Network = nx.erdos_renyi_graph(N, 0.2, seed=2023)
        elif RG_model == "BA":
            Network = nx.barabasi_albert_graph(N, 5, seed=2023)
        else:
            raise NotImplementedError(f"{RG_model} not yet implemented")
        f = plt.figure(100)
        nx.draw(Network)
        f.savefig(f"{RG_model}.pdf", bbox_inches='tight')
        plt.show()

        for p in ps:
            run_ucb(Agents, Network, p, RG_model)
        # def f(p):
        #     return run_ucb(Agents, Network, p, RG_model)
        #
        # with mp.Pool() as pool:
        #     output = pool.map(f, ps)
