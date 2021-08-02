
from src.knowledge_graph.walk_graph import find_segues
from random import sample, choice


def rnd(I, interestingness_weights):
    """Random algorithms, returns a random permutation of the input as a story, with relative segues, picked at random.

    Args:
        I (list): List of KGs representing songs

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    O = sample(I, k=len(I))
    S = []
    for o_1, o_2 in zip(O, O[1:]):
        segues = find_segues(o_1, o_2)
        try:
            S.append(choice(segues))
        except IndexError:
            S.append(None)
    return O, S
