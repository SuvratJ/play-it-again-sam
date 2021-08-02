
from src.knowledge_graph.segue_type import segue_type
from src.knowledge_graph.segue_similarity import segue_similarity
import math


def narrative_strategy_homogeneity_with_decay(s, S, score, decay_constant):
    """Homogeneity narrative strategy with decay.
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are awarded.
    Moreover, this narrative strategy does not look at just one step in the past, but up to 4*decay_constant steps.

    Args:
        s (segue): Last segue to be added to the story
        S (list): The story up to now
        score (float): the score of the segue
        decay_constant

    Returns:
        The score of the segue after this narrative strategy.
    """

    if segue_type(S[-1]) != segue_type(s):
        award = min(sum(math.pow(math.e, -k/decay_constant)*segue_similarity(s, S[-k-1]) for k in range(len(S))), 1)
        score_with_homogeity = score * award
        return score_with_homogeity
    else:
        return 0


def narrative_strategy_diversity_with_decay(s, S, score, decay_constant):
    """Diversity narrative strategy with decay.
    - segues of the same type do not repete themselves;
    - segues of similar type to appear one after the other are penalised.
    Moreover, this narrative strategy does not look at just one step in the past, but up to 4*decay_constant steps.

    Args:
        s (segue): Last segue to be added to the story
        S (list): The story up to now
        score (float): the score of the segue
        decay_constant

    Returns:
        The score of the segue after this narrative strategy.
    """
    if segue_type(S[-1]) != segue_type(s):
        penalisation = max((1-sum(math.pow(math.e, -k/decay_constant)*segue_similarity(s, S[-k-1]) for k in range(len(S)))), 0)
        score_with_diversity = score * penalisation
        return score_with_diversity
    else:
        return 0
