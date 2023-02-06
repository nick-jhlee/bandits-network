

def main_gammas_parallel(T, N, K, Network, p, discards, n_gossips, mps, gammas, repeats=10):
    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []

    # discard, n_gossip, mp, repeats, ps
    params = list(product(discards, n_gossips, mps, range(repeats), gammas))

    def F(param):
        # reseeding!
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # create problem instance
        problem = create_problem(T, N, K, Network, param)

        return run_ucb(problem, p)

    # run the experiments in parallel
    with Pool() as pool:
        everything = pool.map_async(F, params)
        everything = everything.get()

    # collect datas in parallel
    partial_params = list(product(discards, n_gossips, mps))

    def f(partial_param):
        len_gammas = len(gammas)

        total_regret = np.zeros((repeats, len_gammas))
        total_communication = np.zeros((repeats, len_gammas))

        for repeat, i_gamma in product(range(repeats), range(len_gammas)):
            idx = params.index(partial_param + (repeat, gammas[i_gamma]))
            total_regret[repeat][i_gamma] = everything[idx][0]
            total_communication[repeat][i_gamma] = everything[idx][1]

        return total_regret, total_communication

    with Pool() as pool:
        finals = pool.map_async(f, partial_params)
        finals = finals.get()

    for total_regret, total_communication in finals:
        final_regrets_mean.append(np.mean(total_regret, axis=0))
        final_regrets_std.append(np.std(total_regret, axis=0))
        final_communications_mean.append(np.mean(total_communication, axis=0))
        final_communications_std.append(np.std(total_communication, axis=0))

    fname_regret = f"heterogeneous/Regret_final_gamma_p={p}_{Network.name}"
    fname_communication = f"heterogeneous/Communication_final_gamma_p={p}_{Network.name}"

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting
    legends = [f"{mp} (n_gossip={n_gossip}, discard={discard})" for discard, n_gossip, mp in partial_params]
    plot_final(np.array(final_regrets_mean), np.array(final_regrets_std), gammas,
               f"Final Regret ({Network.name}, p={p})", "gamma",
               legends, f"{fname_regret}.pdf")

    plot_final(np.array(final_communications_mean), np.array(final_communications_std), gammas,
               f"Final Communication ({Network.name}, p={p})", "gamma",
               legends, f"{fname_communication}.pdf")


def main_ps(T, N, K, Network, gamma, discards, n_gossips, mps, ps, repeats=10):
    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []
    legends = []

    for discard in discards:
        for n_gossip in n_gossips:
            for mp in mps:
                # update legends
                legends.append(f"{mp} (n_gossip={n_gossip}, discard={discard})")

                # create problem instance
                problem = create_problem(T, N, K, Network, [discard, n_gossip, mp, -1, gamma])

                total_regret = np.zeros((repeats, len(ps)))
                total_communication = np.zeros((repeats, len(ps)))
                for repeat in range(repeats):
                    # reseeding!
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

                    # for parallel loops over p's
                    def f(p):
                        return run_ucb(problem, p)

                    with Pool() as pool:
                        final_p = pool.map_async(f, ps)
                        final_p = final_p.get()

                    for i, item in enumerate(final_p):
                        total_regret[repeat][i] = item[0]
                        total_communication[repeat][i] = item[1]

                final_regrets_mean.append(np.mean(total_regret, axis=0))
                final_regrets_std.append(np.std(total_regret, axis=0))
                final_communications_mean.append(np.mean(total_communication, axis=0))
                final_communications_std.append(np.std(total_communication, axis=0))

    fname_regret = f"heterogeneous/Regret_final_p_gamma={gamma}_{Network.name}"
    fname_communication = f"heterogeneous/Communication_final_p_gamma={gamma}_{Network.name}"

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting
    plot_final(np.stack(final_regrets_mean), np.stack(final_regrets_std), ps,
               f"Final Regret ({Network.name}, gamma={gamma})", "p", legends, f"{fname_regret}.pdf")

    plot_final(np.stack(final_communications_mean), np.stack(final_communications_std), ps,
               f"Final Communication ({Network.name}, gamma={gamma})", "p", legends, f"{fname_communication}.pdf")


def main_gammas(T, N, K, Network, p, discards, n_gossips, mps, gammas, repeats=10):
    final_regrets_mean, final_regrets_std = [], []
    final_communications_mean, final_communications_std = [], []

    for discard in discards:
        for n_gossip in n_gossips:
            for mp in mps:
                total_regret, total_communication = np.zeros((len(gammas), repeats)), np.zeros((len(gammas), repeats))
                for i, gamma in enumerate(gammas):
                    # create problem instance
                    problem = create_problem(T, N, K, Network, [discard, n_gossip, mp, -1, gamma])

                    # reseeding!
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

                    # for parallel loops over repeats
                    def g(repeat):
                        return run_ucb(problem, p=1.0)

                    with Pool() as pool:
                        final_gamma = pool.map_async(g, list(range(repeats)))
                        final_gamma = final_gamma.get()

                    final_regret_gamma, final_communication_gamma = [], []
                    for regret, communication in final_gamma:
                        final_regret_gamma.append(regret)
                        final_communication_gamma.append(communication)

                    total_regret[i] = final_regret_gamma
                    total_communication[i] = final_communication_gamma

                total_regret, total_communication = total_regret.T, total_communication.T
                final_regrets_mean.append(np.mean(total_regret, axis=0))
                final_regrets_std.append(np.std(total_regret, axis=0))
                final_communications_mean.append(np.mean(total_communication, axis=0))
                final_communications_std.append(np.std(total_communication, axis=0))

    final_regrets_mean, final_regrets_std = np.stack(final_regrets_mean), np.stack(final_regrets_std)
    final_communications_mean, final_communications_std = np.stack(final_communications_mean), np.stack(
        final_communications_std)

    fname_regret = f"heterogeneous/Regret_final_gamma_p={p}_{Network.name}"
    fname_communication = f"heterogeneous/Communication_final_gamma_p={p}_{Network.name}"

    # saving final results
    np.savez(f"{fname_regret}.npz", final_regrets_mean, final_regrets_std)
    np.savez(f"{fname_communication}.npz", final_communications_mean, final_communications_std)

    # plotting
    legends = [f"{mp} (n_gossip={n_gossip}, discard={discard})" for discard, n_gossip, mp in
               product(discards, n_gossips, mps)]
    plot_final(np.array(final_regrets_mean), np.array(final_regrets_std), gammas,
               f"Final Regret ({Network.name}, p={p})", "gamma", legends, f"{fname_regret}.pdf")

    plot_final(np.array(final_communications_mean), np.array(final_communications_std), gammas,
               f"Final Communication ({Network.name}, p={p})", "gamma", legends, f"{fname_communication}.pdf")