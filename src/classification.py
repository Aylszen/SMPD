import numpy as np


def calculate_distances(pattern, matrix):
    distances = []
    print("Pattern: ", pattern)
    for m_pattern in matrix:
        print("m_pattern: ", m_pattern)
        distances.append(np.linalg.norm(m_pattern - pattern))
    distances.sort()
    return distances


def nn_classification(training_set):

    return True
