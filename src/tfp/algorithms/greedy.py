from src.knowledge_graph.walk_graph import find_segues
from src.interestingness.interestingness_GB import interestingness
from src.tfp.algorithms.common import narrative_strategy_diversity_with_decay, narrative_strategy_homogeneity_with_decay
from src.knowledge_graph.segue_type import segue_type
from src.knowledge_graph.segue_similarity import segue_similarity
from heapq import *


def greedy_template(I, interestingness_weights, init=lambda I: I[0], narrative_strategy=None):
    O = [init(I)]
    pool = set(I)-set(O)
    S = []

    while pool:

        q = []
        for e in pool:

            segues = find_segues(O[-1], e)
            # push dummy None segue with utility 0, in case I do not find any segue
            heappush(q, (0, (id(e), e, None)))

            for segue in segues:
                score = interestingness([segue], **interestingness_weights)[0]

                # diversity
                score = narrative_strategy(segue, S, score) if len(S) > 0 and narrative_strategy is not None else score

                heappush(q, (-score, (id(segue), e, segue)))

        _, o, s = heappop(q)[1]

        O.append(o)
        S.append(s)
        pool = set(I)-set(O)

    return O, S


def greedy(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    return greedy_template(I, interestingness_weights, init)


def greedy_diversity_binary(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with binary narrative strategy: segues of the same type do not repete themselves.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=lambda s, S, score: score if segue_type(S[-1]) != segue_type(s) else 0)


def greedy_diversity(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with narrative strategy:
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are penalised.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=lambda s, S, score: score*(1-segue_similarity(S[-1], s)))


def greedy_diversity_with_decay_1(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with narrative strategy:
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are penalised.
    Moreover, this narrative strategy does not look at just one step in the past, but up to 4 steps.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    def narrative_strategy(s, S, score):
        narrative_strategy_diversity_with_decay(s, S, score, 1)

    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=narrative_strategy)


def greedy_diversity_with_decay_3(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with narrative strategy:
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are penalised.
    Moreover, this narrative strategy does not look at just one step in the past, but up to 12 steps.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    def narrative_strategy(s, S, score):
        narrative_strategy_diversity_with_decay(s, S, score, 3)

    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=narrative_strategy)


def greedy_homogeneity(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with narrative strategy:
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are awarded.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=lambda s, S, score: score*segue_similarity(S[-1], s) if segue_type(S[-1]) != segue_type(s) else 0)


def greedy_homogeneity_with_decay_1(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with narrative strategy:
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are awarded.
    Moreover, this narrative strategy does not look at just one step in the past, but up to 4 steps.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """

    def narrative_strategy(s, S, score):
        narrative_strategy_homogeneity_with_decay(s, S, score, 1)

    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=narrative_strategy)


def greedy_homogeneity_with_decay_3(I, interestingness_weights, init=lambda I: I[0]):
    """Solve the tfp using the greedy solution, with narrative strategy:
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are awarded.
    Moreover, this narrative strategy does not look at just one step in the past, but up to 12 steps.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """

    def narrative_strategy(s, S, score):
        narrative_strategy_homogeneity_with_decay(s, S, score, 3)

    return greedy_template(I, interestingness_weights, init=init, narrative_strategy=narrative_strategy)
