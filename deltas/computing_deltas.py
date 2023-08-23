from utils_delta import *
from multiprocessing import Pool
from main import *

## set-up
N = 30  # number of agents
K = 10  # total number of arms

tmp = np.sort(np.random.uniform(size=K))
# if not os.path.exists("deltas/means.npz"):
#     tmp = np.sort(np.random.uniform(size=K))
#     np.savez("deltas/means.npz", tmp=tmp)
# else:
#     tmp = np.load("deltas/means.npz")
#     tmp = tmp['tmp']

reward_avgs = {a: tmp[a] for a in range(K)}
total_arm_set = list(range(K))

# uniform arm distribution
arm_sets_uniform = uniform_arm_sets(N, K)
# if not os.path.exists(f"deltas/arm_sets_uniform.json"):
#     arm_sets_uniform = create_uniform_arm_sets(N, K, k)
#     with open(f"deltas/arm_sets_uniform.json", "w") as f:
#         json.dump(arm_sets_uniform, f)
# else:
#     with open(f"deltas/arm_sets_uniform.json", "r") as f:
#         arm_sets_uniform = json.load(f)


## Exp. 1 delta vs. p
## create ER networks of different densities
ER_diameters, ER_components, ER_deltas, ER_regret_gaps = [], [], [], []
u = 2023
gamma = 3
er_ps = np.linspace(0.01, 0.6, 20)
# er_ps = np.linspace(0.01, 2.0, 20) / N ** (0.5 * (1 - (1 / gamma)))
n_repeat = 10
T = int(1e2)

if not os.path.exists("deltas/vary_p/networks"):
    os.makedirs("deltas/vary_p/networks")


def F(Network):
    return compute_delta(Network, arm_sets_uniform, reward_avgs, K, gamma)


def F_regret(Network):
    Agents = [Agent(v, arm_sets_uniform[v], reward_avgs, Network, 0, 0, K) for v in range(N)]

    regret_gap = 0
    for _ in range(10):
        # Flooding
        problem = create_problem(Network, Agents, T, N, K, ("Flooding", None, gamma, 1.0, n_repeat))
        Group_Regrets_flooding, _, _, _ = run_ucb(problem, 1.0)

        # FWA
        problem = create_problem(Network, Agents, T, N, K, ("Flooding-Absorption", None, gamma, 1.0, n_repeat))
        Group_Regrets_fwa, _, _, _ = run_ucb(problem, 1.0)

        regret_gap += Group_Regrets_fwa[-1] - Group_Regrets_flooding[-1]
    return regret_gap
    # for _ in range(n_repeat):
    #     # Flooding
    #     problem = create_problem(Network, Agents, T, N, K, (None, "Flooding", gamma, 1.0, n_repeat))
    #     Group_Regrets_flooding, _, _, _ = run_ucb(problem, 1.0)
    #
    #     # FWA
    #     problem = create_problem(Network, Agents, T, N, K, (None, "Flooding-Absorption", gamma, 1.0, n_repeat))
    #     Group_Regrets_fwa, _, _, _ = run_ucb(problem, 1.0)
    #
    #     regret_gap.append(Group_Regrets_fwa[-1] - Group_Regrets_flooding[-1])
    # return regret_gap


for u in tqdm(range(2023, 2023 + n_repeat)):
    # theoretical bound (compute delta)
    Networks, diameters, components = [], [], []
    for er_p in er_ps:
        Network = gnx.erdos_renyi_graph(N, er_p, seed=u)
        Network.name = f"ER_{er_p}"
        pos = gnx.spring_layout(Network)
        plot_network(Network, pos, parent="deltas/vary_p/networks")
        Networks.append(Network)

        ccs = nx.connected_components(Network)
        components.append(len(list(ccs)))
        if len(list(ccs)) == 0:
            diameters.append(0)
        else:
            diameters.append(nx.diameter(Network.subgraph(max(ccs, key=len))))

    with Pool() as pool:
        finals = pool.map_async(F, Networks)
        finals = finals.get()

    ER_deltas.append(finals)
    ER_diameters.append(diameters)
    ER_components.append(components)

    # # empirical regret gap (T=500)
    # with Pool() as pool:
    #     finals = pool.map_async(F_regret, ER_Networks)
    #     finals = finals.get()
    #
    # ER_regret_gaps.append(finals)

names = ["delta", "diameter", "component", "regret_gap"]
for i, ER_i in enumerate([ER_deltas, ER_diameters, ER_components]):
    # for i, ER_i in enumerate([ER_deltas, ER_regret_gaps]):
    ER_i = np.array(ER_i)
    ER_i_means = np.mean(ER_i, axis=0)
    ER_i_stds = np.std(ER_i, axis=0)

    fname = f"vary_p_{names[i]}"

    np.savez(f"deltas/{fname}.npz", ER_deltas_means=ER_i_means, ER_deltas_stds=ER_i_stds, er_ps=er_ps)

    plt.figure(i)
    plt.errorbar(er_ps, ER_i_means, ER_i_stds)
    plt.title(f"ER, {names[i]} vs. p (gamma={gamma})")
    plt.xlabel("p")
    plt.ylabel(names[i])
    plt.savefig(f"deltas/{fname}.pdf", dpi=500)
plt.show()