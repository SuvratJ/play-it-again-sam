
from src.knowledge_graph.walk_graph import find_segues
from src.interestingness.interestingness_GB import interestingness
from heapq import *


def circular_greedy(I, interestingness_weights, init=lambda I: I[0]):
    """Always greedy, but we move away from the seed song once expanding backwards and the other time forwards.

    Args:
        I (list): List of KGs representing songs
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    # Seed song is taken as the first song. It might as well be taken as random
    O = [init(I)]
    pool = set(I)-set(O)
    S = []

    while pool:

        q = []
        # push dummy None segue with utility 0, in case I do not find any segue
        heappush(q, (0, (id(list(pool)[0]), list(pool)[0], None)))

        for e in pool:

            segues = find_segues(O[-1], e) if len(pool) % 2 == 0 else find_segues(e, O[0])
            for segue in segues:
                score = interestingness([segue], **interestingness_weights)[0]
                heappush(q, (-score, (id(segue), e, segue)))

        _, o, s = heappop(q)[1]

        O.append(o) if len(pool) % 2 == 0 else O.insert(0, o)
        S.append(s) if len(pool) % 2 == 0 else S.insert(0, s)
        pool = set(I)-set(O)

    return O, S
