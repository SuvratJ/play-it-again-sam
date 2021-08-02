"""
Created on Mon Jan 11 2021

@author Giovanni Gabbolini
"""

from src.data.data import preprocessed_dataset_path
from src.knowledge_graph.io import load_sub_graphs_generator
from src.interestingness.interestingness_GB import interestingness
from src.tfp.algorithms import *
import matplotlib.ticker as ticker
from src.interestingness.interestingness_GB import best_interestingness_weights
import numpy as np
from tqdm import tqdm
from cycler import cycler
import matplotlib.pyplot as plt
plt.style.use('science')


def plot():
    font_size = 9
    func = np.average
    l = np.load(f"{preprocessed_dataset_path}/tfp/performance/fine_tune_hill_climbing.npy", allow_pickle=True).item()
    patience_values = [0, 1, 2, 3, 4, 6, 8, 10]
    x = []
    for k in tqdm(patience_values[::-1]):
        v = l[k]
        # Update the array with solution for every restart, with the best solution found up to every restart.
        # Notice that the value of the solution is measured by the sum of interestingness, compute with best weights!
        v = np.array(v)
        sums = np.copy(v)
        for i in range(sums.shape[0]):
            sums[i, 0] = np.sum(v[i, 0][v[i, 0] != -np.inf])
            for j in range(1, sums.shape[1]):
                sums[i, j] = np.sum(v[i, j][v[i, j] != -np.inf])
                if sums[i, j] < sums[i, j-1]:
                    sums[i, j] = sums[i, j-1]
                    v[i, j] = v[i, j-1]
        a = np.zeros(sums.shape)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = func(v[i, j][v[i, j] != -np.inf])
        a = np.average(a, axis=0)
        x.append((a, k))

    monochrome = (cycler('color', ['black', 'grey']) * cycler('marker', ['']) *
                  cycler('linestyle', ['-', '--', ':', '-.']))
    plt.rc('axes', prop_cycle=monochrome)
    ax = plt.subplot()
    for a, k in x:
        plt.plot(a, label=f"Patience = {k}")
    plt.legend(bbox_to_anchor=(1, 1), loc=2, fontsize=font_size)
    ax.grid(which='both', linewidth=.2)
    ax.set_xticks(range(0, 1100, 100))
    ax.set_xticklabels([str(i) for i in range(0, 11, 1)])
    loc = ticker.MultipleLocator(base=.01)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax.set_xlabel("Restarts / 100", fontsize=font_size)
    ax.set_ylabel("$utility$", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.savefig(f"{preprocessed_dataset_path}/tfp/performance/hc", bbox_inches='tight', dpi=400)


def algorithm(I, patience, d_segues):
    return hill_climbing_template(I, best_interestingness_weights(), n_restarts=1000, patience=patience, return_solutions_for_all_restarts=True, d_segues=d_segues)


def _save():
    patience_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = {k: [] for k in patience_values}

    for playlist_reader in tqdm(load_sub_graphs_generator(f"tfp/side")):
        songs = playlist_reader()
        d_segues = {}
        for patience in patience_values:
            solutions = algorithm(songs, patience, d_segues)
            v = [np.array(interestingness(segues, **best_interestingness_weights())) for _, segues in solutions]
            results[patience].append(v)

    try:
        l = np.load(f"{preprocessed_dataset_path}/tfp/performance/fine_tune_hill_climbing.npy", allow_pickle=True).item()
        for k, v in results.items():
            l[k] = v
    except IOError:
        l = results

    np.save(f"{preprocessed_dataset_path}/tfp/performance/fine_tune_hill_climbing", l)


if __name__ == "__main__":
    _save()
    plot()
