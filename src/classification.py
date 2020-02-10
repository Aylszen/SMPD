import numpy as np


def calculate_distances(pattern, matrix):
    distances = []
    for m_pattern in matrix:
        distances.append(np.linalg.norm(m_pattern - pattern))
    distances.sort()
    return distances

def assign_to_set(distances_matrix_a, distances_matrix_b, training_set):
    return False


def nn_classification(training_set, matrix_after_sel_alg):
    distances_matrix_a = []
    distances_matrix_b = []
    for training_char in training_set:
        distances_matrix_a.append(calculate_distances(training_char, matrix_after_sel_alg[0]))
        distances_matrix_b.append(calculate_distances(training_char, matrix_after_sel_alg[1]))

    print("A", distances_matrix_a)
    print("B", distances_matrix_b)
    return True
