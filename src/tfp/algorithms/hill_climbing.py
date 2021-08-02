
from src.knowledge_graph.walk_graph import find_segues
from src.interestingness.interestingness_GB import interestingness
from random import sample, choice
import numpy as np


def faster_interestingness(segue, interestingness_weights):
    if segue is None:
        return 0
    else:
        try:
            return_value = segue['interestingness']
        except KeyError:
            i = interestingness([segue], **interestingness_weights)[0]
            segue['interestingness'] = i
            return_value = i
        return return_value


def faster_find_segues(e_1, e_2, d_segues):
    try:
        return_value = d_segues[(e_1[0], e_2[0])]
    except KeyError:
        return_value = find_segues(e_1[1], e_2[1])
        d_segues[(e_1[0], e_2[0])] = return_value
    return return_value


def initialise_solution(I, d_segues):
    O = sample(I, k=len(I))
    S = []
    for o_1, o_2 in zip(O, O[1:]):
        segues = faster_find_segues(o_1, o_2, d_segues)
        try:
            S.append(choice(segues))
        except IndexError:
            S.append(None)
    return O, S


def hill_climbing_template(I, interestingness_weights, n_restarts=1, patience=0, return_solutions_for_all_restarts=False, d_segues=None):
    # This method implements a Steepest Ascent Hill Climbing Algorithm, with configurable numbers of restarts and sideways moves allowed.

    # Objective function
    def obj(S):
        scores = np.zeros(len(S))
        for idx in range(len(S)):
            score = faster_interestingness(S[idx], interestingness_weights)
            scores[idx] = score
        return np.sum(scores)

    # Write original position of elements in I, now I is a list [(0, I[0]), ... , (n, I[n])]
    I = [e for e in enumerate(I)]
    # Dictionary indixed by the original position of elements in I, that holds all the segues between that couple.
    d_segues = {} if d_segues is None else d_segues

    solutions = [None]*n_restarts

    for n_restart in range(n_restarts):
        actual_patience = patience

        # Initialise the solution at random.
        O, S = initialise_solution(I, d_segues)

        # O and S store the actual solution we are working on, that might be suboptimal, e.g. in case we did some sideways moves.
        # best_O and best_S store the best solution.
        best_O = O
        best_S = S

        while True:
            # Find neighborhood.
            swap_idx_1 = choice(range(1, len(O)-2))
            swap_idx_2 = swap_idx_1 + 1

            if S[swap_idx_1] is None:
                middle_segue = None
            else:
                middle_segue = {
                    'n1': S[swap_idx_1]['n2'],
                    'n2': S[swap_idx_1]['n1'],
                    'value': S[swap_idx_1]['value'],
                    'compare_function': S[swap_idx_1]['compare_function'],
                }
                if 'interestingness' in S[swap_idx_1]:
                    middle_segue['interestingness'] = S[swap_idx_1]['interestingness']

            segues_1 = faster_find_segues(O[swap_idx_1-1], O[swap_idx_2], d_segues)
            segues_2 = faster_find_segues(O[swap_idx_1], O[swap_idx_2+1], d_segues)
            segues_1 = [None] if not len(segues_1) else segues_1
            segues_2 = [None] if not len(segues_2) else segues_2

            # Find best neighbor.
            # We do not evaluate the obj for every neighbor, as the neighbors grows quadratically.
            # Instead we find the best neighbors as the two segues that alone have higher interestingess.
            best_neighbor_S = []

            scores_1 = [faster_interestingness(segue, interestingness_weights) for segue in segues_1]
            segues_1 = sorted(segues_1, key=lambda segue: -scores_1[segues_1.index(segue)])
            scores_2 = [faster_interestingness(segue, interestingness_weights) for segue in segues_2]
            segues_2 = sorted(segues_2, key=lambda segue: -scores_2[segues_2.index(segue)])
            best_neighbor_S = S[0:swap_idx_1-1]+[segues_1[0], middle_segue, segues_2[0]]+S[swap_idx_2+1:]

            # Update the actual solution with the best neighbor.
            S = best_neighbor_S
            O = O[0:swap_idx_1]+[O[swap_idx_2], O[swap_idx_1]]+O[swap_idx_2+1:]

            # If    the actual solution is better than the best solution, then update the best solution, restore actual patience to patience and loop;
            # Elif  the actual solution is worse or equal than the best solution, then:
            #      If   we still have actual patience decrease actual patience and loop;
            #      Else break

            if obj(S) > obj(best_S):
                best_S = S
                best_O = O
                actual_patience = patience
                continue
            else:
                if actual_patience > 0:
                    actual_patience -= 1
                    continue
                else:
                    break

        solutions[n_restart] = ([e for _, e in best_O], best_S)

    if return_solutions_for_all_restarts:
        return_value = solutions
    else:
        solutions = sorted(solutions, key=lambda solution: obj(solution[1]), reverse=True)
        return_value = solutions[0]

    return return_value


def hill_climbing(I, interestingness_weights):
    """Parameters found after experimentation.
    This is the best configuration in terms of average interestingness,
    which performs great also in terms of std, min, max.

    The n_restarts was set by picking a number after the elbow.
    After 100, the gain in performance is limited.
    """
    return hill_climbing_template(I, interestingness_weights, n_restarts=40,  patience=10)
