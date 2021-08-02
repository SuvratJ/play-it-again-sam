"""
Created on Tue Apr 13 2021

@author Giovanni Gabbolini
"""


from src.knowledge_graph.walk_graph import find_segues
import tempfile
import os
import numpy as np
from src.interestingness.interestingness_GB import interestingness
from concorde.tsp import TSPSolver


def optimal(I, interestingness_weights):
    """Finds the optimal tour of songs that maximise interestingness.

    It works using the Concorde TSP tool, https://github.com/jvkersch/pyconcorde.

    Our problem can be solved as a TSP, infact our problems resembles the optimal hamiltonian path problem.
    And, the hamiltonian path problem can be solved with TSP solvers by introducing a dummy source node.

    Args:
        I (list): List of KGs representing songs.
        init (func): Selects the song from which the O should start.

    Returns:
        Two lists O and S with KGs representing songs and segues
    """
    # build reward matrix based on interestingness of going from one song to another.
    matrix = np.zeros((len(I), len(I)))
    for i in range(len(I)):
        for j in range(len(I)):

            if i == j:
                matrix[i, j] = 0.0

            segues = find_segues(I[i], I[j])
            scores = interestingness(segues, **interestingness_weights)
            matrix[i, j] = max(scores) if len(scores) else 0.0

    # matrix might be not symmetric due to sampled interestingness. we force it to be so.
    matrix = (matrix + matrix.T)/2

    # transform matrix from a reward matrix to a distance matric, the TSP solver minimises.
    # high interestingness -> low distance.
    matrix = 1-matrix

    # add fake dummy node to solve optimal hamiltonian path as a TSP.
    matrix = np.vstack((np.zeros((1, matrix.shape[1])), matrix))
    matrix = np.hstack((np.zeros((matrix.shape[0], 1)), matrix))

    # convert distances from floats to integers from 0 to 100.
    matrix = (100*matrix).astype(np.int)

    # write the problem in a temporary file used by the concorde tool
    ccdir = tempfile.mkdtemp()
    ccfile = os.path.join(ccdir, 'data.tsp')
    with open(ccfile, 'w') as fp:
        fp.write(f"TYPE: TSP\nDIMENSION: {matrix.shape[0]}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        fp.write(np.array2string(matrix, threshold=np.inf).replace('[', '').replace(']', '').replace('\n ', '\n'))
        fp.write("\nEOF")

    # solve using concorde solver.
    solver = TSPSolver.from_tspfile(ccfile)
    solution = solver.solve().tour
    solution = [i-1 for i in solution[1:]]

    # build output.
    O = [I[idx] for idx in solution]
    S = []
    for s_1, s_2 in zip(O, O[1:]):
        segues = find_segues(s_1, s_2)
        scores = interestingness(segues, **interestingness_weights)
        segues = sorted(segues, key=lambda segue: -scores[segues.index(segue)])
        S.append(segues[0] if len(segues) else None)
    return O, S
