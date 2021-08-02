from src.tfp import algorithms
from src.data.data import preprocessed_dataset_path
from src.utils.experiments import trunc
from collections import defaultdict
import numpy as np
import pandas as pd


def intra_story_diversity(algorithms):
    """Saves a report on the inter story diversity performance of algorithms.

    The inter story diversity is the extent to which algorithms produce different stories, when they start from different seed songs.

    We measure diversity by resorting to Jaccard index on the segue types composing up the stories.
    The result is the percentage of segues of different type in two stories produced with the same songs, but from a different seed.

    Args:
        algorithms (list): functions implementing algorithms for interestingness. It is required to have an 'init' param.
    """
    segues_types = np.load(f"{preprocessed_dataset_path}/tfp/inter_story_diversity/segues_types.npy", allow_pickle=True).item()
    results = {alg.__name__: defaultdict(list) for alg in algorithms}
    for alg in algorithms:
        for t in segues_types[alg.__name__]['0']:
            diversity = len(set(t))/len(t)

            if len(t) > 2 and len(t) <= 10:
                results[alg.__name__]['short_stories'].append(diversity)
            elif len(t) > 10 and len(t) <= 25:
                results[alg.__name__]['mid_stories'].append(diversity)
            else:
                results[alg.__name__]['long_stories'].append(diversity)

    df = pd.DataFrame({
        'short_stories': [trunc(np.average(results[alg.__name__]['short_stories']), 2) for alg in algorithms],
        'mid_stories': [trunc(np.average(results[alg.__name__]['mid_stories']), 2) for alg in algorithms],
        'long_stories': [trunc(np.average(results[alg.__name__]['long_stories']), 2) for alg in algorithms],
        'overall': [trunc(np.average(results[alg.__name__]['long_stories']+results[alg.__name__]['mid_stories']+results[alg.__name__]['short_stories']), 2) for alg in algorithms]
    }, index=[alg.__name__ for alg in algorithms])
    df.to_csv(f"{preprocessed_dataset_path}/tfp/intra_story_diversity/results.csv")


if __name__ == "__main__":
    algos = [random, greedy, greedy_diversity_binary, greedy_diversity,
             greedy_diversity_with_decay_1, greedy_diversity_with_decay_3, greedy_homogeneity, greedy_homogeneity_with_decay_1, greedy_homogeneity_with_decay_3]
    intra_story_diversity(algos)
