"""
Created on Mon Jan 11 2021

@author Giovanni Gabbolini
"""

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.tfp.algorithms import *
from src.tfp.offline_experiments.utils import *
from src.data.data import preprocessed_dataset_path
from src.knowledge_graph.segue_type import segue_type
from src.utils.experiments import trunc
from src.knowledge_graph.io import load_sub_graphs_generator
from src.interestingness.interestingness_GB import interestingness
from src.interestingness.interestingness_GB import best_interestingness_weights
plt.style.use('science')


def diversity_table(algorithms):

    def diversity(d, algorithm_name, f):
        l = d[algorithm_name]
        l = [e for e in l if f(e)]
        a = [len(set(e))/len(e) for e in l]
        return np.average(a)

    d = np.load(f"{preprocessed_dataset_path}/tfp/performance/segue_types.npy", allow_pickle=True).item()
    r = {
        'short_stories': [trunc(diversity(d, algorithm.__name__, lambda l: len(l) < 10), 3) for algorithm in algorithms],
        'mid_stories': [trunc(diversity(d, algorithm.__name__, lambda l: len(l) >= 10 and len(l) < 25), 3) for algorithm in algorithms],
        'long_stories': [trunc(diversity(d, algorithm.__name__, lambda l: len(l) >= 25), 3) for algorithm in algorithms],
    }
    df_diversity = pd.DataFrame(r, index=[a.__name__ for a in algorithms])
    df_diversity.to_csv(f"{preprocessed_dataset_path}/tfp/performance/df_diversity.csv")


def average_interestingness_table(algorithms):

    def average_interestingness(d, algorithm_name, f):
        l = d[algorithm_name]
        l = [np.array(e) for e in l if f(e)]
        a = [np.average(e[e != -np.inf]) for e in l]
        return np.average(a)

    d = np.load(f"{preprocessed_dataset_path}/tfp/performance/interestingness_scores.npy", allow_pickle=True).item()
    r = {
        'short_stories': [trunc(average_interestingness(d, algorithm.__name__, lambda l: len(l) < 10), 3) for algorithm in algorithms],
        'mid_stories': [trunc(average_interestingness(d, algorithm.__name__, lambda l: len(l) >= 10 and len(l) < 25), 3) for algorithm in algorithms],
        'long_stories': [trunc(average_interestingness(d, algorithm.__name__, lambda l: len(l) >= 25), 3) for algorithm in algorithms],
    }
    df_average_interestingness = pd.DataFrame(r, index=[a.__name__ for a in algorithms])
    df_average_interestingness.to_csv(f"{preprocessed_dataset_path}/tfp/performance/df_average_interestingness.csv")


def plot(algorithms, func=np.average, legend=True, title=None, x_axis_label=None, y_axis_label=None):
    """Compute performance in interestingness according to a function `func` as story length increases.

    Args:
        func (callable, optional): Defaults to np.average. Examples: np.max, np.min, np.std.
    """
    font_size = 22
    ax = plt.subplot()
    d = np.load(f"{preprocessed_dataset_path}/tfp/performance/interestingness_scores.npy", allow_pickle=True).item()
    for a in algorithms:
        x = []
        y = []
        for l in d[a.__name__]:
            x.append(len(l)+1)
            l = [e for e in l if e != -np.inf]
            y.append(func(l))
        x, y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x, y) if xVal == a])) for xVal in set(x)))
        x, y = np.array(x), np.array(y)
        # plot the data
        # style = line_style(a)
        # style['color'] = lighten_color(style['color'], 0.4)
        # plt.plot(x, y, linewidth=0.40, **line_style(a))
        # also plot an line fitting of the data
        order = 3
        p = np.polyfit(x, y, order)
        line = np.zeros((order+1, len(x)))
        for i in range(order+1):
            line[i, :] = p[i] * (x**(order-i))
        line = np.sum(line, axis=0)
        plt.plot(x, line, label=label(a), **line_style(a), linewidth=2)
    if x_axis_label:
        ax.set_xlabel(x_axis_label, fontsize=font_size)
    if y_axis_label:
        ax.set_ylabel(y_axis_label, fontsize=font_size)
    if legend:
        plt.legend(fontsize=font_size, bbox_to_anchor=(1.05, 1), loc='upper left')
    if title:
        plt.title(title, fontsize=font_size)
    ax.grid(True, which='both', linewidth=0.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.2f'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.savefig(f"{preprocessed_dataset_path}/tfp/performance/{func.__name__}_plot", bbox_inches='tight', dpi=400)
    ax.clear()


def _save(algorithms):
    """Saves a record of the output on the algorithms, in two qualities:

    * Interestingness score of the segues building up stories
    * Segues types of the segues building up stories

    Args:
        algorithms (list): functions implementing algorithms for interestingness.
    """
    interestingness_scores = {algorithm.__name__: [] for algorithm in algorithms}
    segue_types = {algorithm.__name__: [] for algorithm in algorithms}
    for playlist_reader in tqdm(load_sub_graphs_generator(f"tfp/main")):
        songs = playlist_reader()
        for algorithm in algorithms:
            _, segues = algorithm(songs, best_interestingness_weights())
            interestingness_scores[algorithm.__name__].append(interestingness(segues, **best_interestingness_weights()))
            segue_types[algorithm.__name__].append([segue_type(s) for s in segues])

    for file_name, var in [('interestingness_scores', interestingness_scores), ('segue_types', segue_types)]:

        try:
            old = np.load(f"{preprocessed_dataset_path}/tfp/performance/{file_name}.npy", allow_pickle=True).item()
            for k, v in var.items():
                old[k] = v
            to_save = old
        except Exception:
            to_save = var

        np.save(f"{preprocessed_dataset_path}/tfp/performance/{file_name}", to_save)


if __name__ == "__main__":
    algos = [optimal, hill_climbing, greedy, ]
    plot(algos, np.average, legend=False, x_axis_label='$|I|$', y_axis_label='$score$', title="(a) Average")
    plot(algos, np.std, legend=False, x_axis_label='$|I|$', title="(b) Standard deviation")
    plot(algos, max, legend=False, x_axis_label='$|I|$', title="(c) Maximum")
    plot(algos, min, legend=True, x_axis_label='$|I|$', title="(d) Minimum")
