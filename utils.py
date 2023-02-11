import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(Regrets, Group_Regrets, Network, titles, fnames):
    # agent-wise quantity
    plt.figure(1)
    order = np.argsort([Network.degree(v) for v in Network])
    for v in Network:
        plt.plot(Regrets[order[v]])
    plt.title(titles[0])
    # sort the legend by degree
    plt.legend([f"Agent {order[v]} (deg = {Network.degree(order[v])})" for v in Network],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(fnames[0], dpi=1200, bbox_inches='tight')
    # plt.show()

    # group-wise quantity
    plt.figure(2)
    plt.plot(Group_Regrets)
    plt.title(titles[1])
    plt.savefig(fnames[1], dpi=1200, bbox_inches='tight')
    # plt.show()


# source: https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans
def plot_final(final_means, final_stds, xs, title, xlabel, legends, fname):
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", len(legends))

    with sns.axes_style("darkgrid"):
        for i, color in enumerate(clrs):
            ax.plot(xs, final_means[i], label=legends[i], c=color)
            ax.fill_between(xs, final_means[i] - final_stds[i], final_means[i] + final_stds[i],
                            alpha=0.3, facecolor=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(fname, dpi=1200, bbox_inches='tight')
    # plt.show()
