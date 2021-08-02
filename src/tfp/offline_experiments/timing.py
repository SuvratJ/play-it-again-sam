import cProfile
import pstats
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data import preprocessed_dataset_path
from src.knowledge_graph.io import load_sub_graphs_generator
from src.interestingness.interestingness_GB import interestingness
from src.tfp.algorithms import *
from src.interestingness.interestingness_GB import best_interestingness_weights
from src.knowledge_graph.segue_type import segue_type
from src.tfp.offline_experiments.utils import *
plt.style.use('science')


def timing_plot(algorithms):
    """It produces a plot that show the time that takes to the algorithm to produce a story, as a function of story length

    Args:
        algorithms (list): functions implementing algorithms for interestingness.
    """
    font_size = 9
    d = np.load(f"{preprocessed_dataset_path}/tfp/performance/timing.npy", allow_pickle=True).item()
    timing, algorithm, story_length = ([] for _ in range(3))
    for a in algorithms:
        algorithm = a.__name__.replace('_', ' ')
        timing = [e[0] for e in d[a.__name__]]
        # +1 as story length is the number of songs, i.e. number of segues + 1
        story_length = [e[1]+1 for e in d[a.__name__]]
        story_length, timing = zip(
            *sorted((xVal, np.mean([yVal for a, yVal in zip(story_length, timing) if xVal == a])) for xVal in set(story_length)))
        story_length, timing = np.array(story_length), np.array(timing)
        order = 3
        p = np.polyfit(story_length, timing, order)
        line = np.zeros((order+1, len(story_length)))
        for i in range(order+1):
            line[i, :] = p[i] * (story_length**(order-i))
        line = np.sum(line, axis=0)
        plt.plot(story_length, line, label=label(a), **line_style(a))
    # df = pd.DataFrame({'Runtime (seconds)': timing, 'algorithm': algorithm, 'Sequence length': story_length})
    # sns.lineplot(data=df, x='Sequence length', y='Runtime (seconds)', hue='algorithm', ci=None)
    plt.yscale('log')
    plt.xlabel('$|I|$', fontsize=font_size)
    plt.ylabel('Second', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True, which='both', linewidth=0.2)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.savefig(f"{preprocessed_dataset_path}/tfp/performance/timing_plot", bbox_inches='tight', dpi=400)


def _save(algorithms):
    # excecute an algorithm once so that preliminary operation are carried out
    [algorithm(load_sub_graphs_generator(f"tfp")[0](), best_interestingness_weights()) for algorithm in algorithms]
    timing = {algorithm.__name__: [] for algorithm in algorithms}
    for playlist_reader in tqdm(load_sub_graphs_generator(f"tfp/main")):
        songs = playlist_reader()
        for algorithm in algorithms:

            with cProfile.Profile() as pr:
                _, segues = algorithm(songs, best_interestingness_weights())
            ps = pstats.Stats(pr).get_stats_profile()
            try:
                timing_interestingness = ps.func_profiles['interestingness'].cumtime
            except KeyError:
                timing_interestingness = 0
            timing[algorithm.__name__].append((timing_interestingness + ps.func_profiles['find_segues'].cumtime, len(segues)))

    try:
        old = np.load(f"{preprocessed_dataset_path}/tfp/performance/timing.npy", allow_pickle=True).item()
        for k, v in timing.items():
            old[k] = v
        to_save = old
    except Exception:
        to_save = timing

    np.save(f"{preprocessed_dataset_path}/tfp/performance/timing", to_save)


if __name__ == "__main__":
    algorithms = [optimal, hill_climbing, greedy]
    # _save(algorithms)
    timing_plot(algorithms)
