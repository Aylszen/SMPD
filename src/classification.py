import numpy as np


def calculate_distances(pattern, matrix):
    distances = []
    for m_pattern in matrix:
        distances.append(np.linalg.norm(m_pattern - pattern))
    distances.sort()
    return distances


def classify_one_best_characteristic(distances_matrix_a, distances_matrix_b, size):
    indexes_list = []
    for i in range(0, size):
        if distances_matrix_a[i][0] > distances_matrix_b[i][0]:
            indexes_list.append(0)
        elif distances_matrix_b[i][0] > distances_matrix_a[i][0]:
            indexes_list.append(1)
        else:
            indexes_list.append(2)

    print(len(indexes_list))
    print(indexes_list)
    return indexes_list


def classify_k_best_characteristics(distances_matrix_a, distances_matrix_b, size, k):
    indexes_list = []
    for i in range(0, size):
        if distances_matrix_a[i][0] > distances_matrix_b[i][0]:
            indexes_list.append(0)
        elif distances_matrix_b[i][0] > distances_matrix_a[i][0]:
            indexes_list.append(1)
        else:
            indexes_list.append(2)

    print(len(indexes_list))
    print(indexes_list)
    return indexes_list


def nn_classification(training_set, matrix_after_sel_alg):
    distances_matrix_a = []
    distances_matrix_b = []
    for training_char in np.transpose(training_set[0]):
        distances_matrix_a.append(calculate_distances(training_char, np.transpose(matrix_after_sel_alg[0])))
        distances_matrix_b.append(calculate_distances(training_char, np.transpose(matrix_after_sel_alg[1])))

    for training_char in np.transpose(training_set[1]):
        distances_matrix_a.append(calculate_distances(training_char, np.transpose(matrix_after_sel_alg[0])))
        distances_matrix_b.append(calculate_distances(training_char, np.transpose(matrix_after_sel_alg[1])))

    length_training_set = len(np.transpose(training_set[0])) + len(np.transpose(training_set[1]))

    return classify_one_best_characteristic(distances_matrix_a, distances_matrix_b, length_training_set)


def calculate_efficiency(classification_result, matrices_all, training_set):
    efficiency = 0
    training_set_a_t = training_set[0].transpose()
    training_set_b_t = training_set[1].transpose()

    for i in range(0, len(training_set_a_t)):
        if classification_result[i] != 1:
            if training_set_a_t[i] in matrices_all[0]:
                efficiency = efficiency + 1
        elif classification_result[i + len(training_set_a_t)] == 2:
            efficiency = efficiency + 1

    for i in range(0, len(training_set_b_t)):
        if classification_result[i + len(training_set_a_t)] == 1:
            if training_set_b_t[i] in matrices_all[1]:
                print("Weszlo4")
                efficiency = efficiency + 1
        elif classification_result[i + len(training_set_a_t)] == 2:
            efficiency = efficiency + 1

    training_set_len = len(training_set[0].transpose()) + len(training_set[1].transpose())
    print(training_set_len)
    print("Len training set: ", len(training_set[0]))
    print("Efficiency: ", efficiency)
    return efficiency/training_set_len * 100