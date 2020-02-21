import random

import numpy as np
from copy import deepcopy


def calculate_mean(matrix, axis):
    return matrix.mean(axis)


def calculate_distances(pattern, matrix):
    distances = []
    for m_pattern in matrix:
        distances.append(np.linalg.norm(m_pattern - pattern))
    distances.sort()
    return distances


def classify_k_best_characteristics(distances_matrix_a, distances_matrix_b, size, k):
    indexes_list = []
    for i in range(0, size):
        temp_val = [0, 0, 0]
        for j in range(0, k):
            if distances_matrix_a[i][0] < distances_matrix_b[i][0]:
                temp_val[0] = temp_val[0] + 1
                del distances_matrix_a[i][0]
            elif distances_matrix_b[i][0] < distances_matrix_a[i][0]:
                temp_val[1] = temp_val[1] + 1
                del distances_matrix_b[i][0]
            else:
                temp_val[2] = temp_val[2] + 1
                del distances_matrix_a[i][0]

        if temp_val[0] > temp_val[1] and temp_val[0] > temp_val[2]:
            indexes_list.append(0)
        elif temp_val[1] > temp_val[0] and temp_val[1] > temp_val[2]:
            indexes_list.append(1)
        else:
            indexes_list.append(2)

    return indexes_list


def nn_classification(training_set_after_sel_alg, matrix_after_sel_alg, matrices_list, training_set):
    return knn_classification(training_set_after_sel_alg, matrix_after_sel_alg, matrices_list, 1, training_set)


def knn_classification(training_set_after_sel_alg, learning_set_after_sel_alg, matrices_list, k, training_set):
    distances_matrix_a = []
    distances_matrix_b = []
    for training_char in np.transpose(training_set_after_sel_alg[0]):
        distances_matrix_a.append(calculate_distances(training_char, np.transpose(learning_set_after_sel_alg[0])))
        distances_matrix_b.append(calculate_distances(training_char, np.transpose(learning_set_after_sel_alg[1])))

    for training_char in np.transpose(training_set_after_sel_alg[1]):
        distances_matrix_a.append(calculate_distances(training_char, np.transpose(learning_set_after_sel_alg[0])))
        distances_matrix_b.append(calculate_distances(training_char, np.transpose(learning_set_after_sel_alg[1])))

    length_training_set = len(np.transpose(training_set_after_sel_alg[0])) + len(np.transpose(training_set_after_sel_alg[1]))

    indexes = classify_k_best_characteristics(distances_matrix_a, distances_matrix_b, length_training_set, k)
    return calculate_efficiency(indexes, matrices_list, training_set)


def nm_classification(training_set_after_sel_alg, learning_set_after_sel_alg, matrices_list, training_set):
    learning_set_matrix_a = calculate_mean(np.transpose(learning_set_after_sel_alg[0]), 0)
    learning_set_matrix_b = calculate_mean(np.transpose(learning_set_after_sel_alg[1]), 0)
    print(len(learning_set_matrix_a))
    distances_matrix_a = []
    distances_matrix_b = []
    for training_char in np.transpose(training_set_after_sel_alg[0]):
        distances_matrix_a.append(calculate_distance_nm(training_char, learning_set_matrix_a))
        distances_matrix_b.append(calculate_distance_nm(training_char, learning_set_matrix_b))

    for training_char in np.transpose(training_set_after_sel_alg[1]):
        distances_matrix_a.append(calculate_distance_nm(training_char, learning_set_matrix_a))
        distances_matrix_b.append(calculate_distance_nm(training_char, learning_set_matrix_b))

    length_training_set = len(np.transpose(training_set_after_sel_alg[0])) + len(np.transpose(training_set_after_sel_alg[1]))

    indexes = classify_k_best_characteristics(distances_matrix_a, distances_matrix_b, length_training_set, 1)
    print(indexes)
    return calculate_efficiency(indexes, matrices_list, training_set)


def calculate_efficiency(classification_result, matrices_all, training_set):
    efficiency = 0
    training_set_a_t = training_set[0].transpose()
    training_set_b_t = training_set[1].transpose()

    for i in range(0, len(training_set_a_t)):
        if classification_result[i] == 0:
            if training_set_a_t[i] in matrices_all[0]:
                efficiency = efficiency + 1
        elif classification_result[i] == 2:
            efficiency = efficiency + 1

    for i in range(0, len(training_set_b_t)):
        if classification_result[i + len(training_set_a_t)] == 1:
            if training_set_b_t[i] in matrices_all[1]:
                efficiency = efficiency + 1
        elif classification_result[i + len(training_set_a_t)] == 2:
            efficiency = efficiency + 1

    training_set_len = len(training_set[0].transpose()) + len(training_set[1].transpose())

    return efficiency/training_set_len * 100


def calculate_distance_nm(pattern, matrix):
    distances = []
    distances.append(np.linalg.norm(pattern - matrix))
    return distances


def calculate_distances_knm(pattern, matrix):
    distances = []
    for m_pattern in matrix:
        distances.append(np.linalg.norm(m_pattern - pattern))
    return distances


def randomly_choose_sample(learning_set_after_sel_alg):
    mean_list = []
    index = random.randint(0, len(np.transpose(learning_set_after_sel_alg[0])) - 1)
    mean_list.append(np.transpose(learning_set_after_sel_alg[0])[index])
    index = random.randint(0, len(np.transpose(learning_set_after_sel_alg[1])) - 1)
    mean_list.append(np.transpose(learning_set_after_sel_alg[1])[index])
    if np.array_equal(mean_list[0], mean_list[1]):
        print("Weszlo!!")
        mean_list = randomly_choose_sample(learning_set_after_sel_alg)
    return mean_list


def knm_classification(training_set_after_sel_alg, learning_set_after_sel_alg, matrices_list, k, training_set):
    mean_list = randomly_choose_sample(learning_set_after_sel_alg)
    new_learning_set = deepcopy(learning_set_after_sel_alg[0])
    new_learning_set = np.append(new_learning_set, learning_set_after_sel_alg[1], axis=1)

    return knm_evaluate_sets(np.transpose(new_learning_set), mean_list, training_set_after_sel_alg, matrices_list, training_set)


def knm_evaluate_sets(learning_set, mean_list, training_set_after_sel_alg, matrices_list, training_set):
    matrix_a, matrix_b, new_mean_list = knm_evaluate_sets_first_step(learning_set, mean_list)
    new_matrix_a, new_matrix_b = knm_evaluate_sets_second_step(matrix_a, matrix_b, new_mean_list, learning_set)
    new_learning_set = []
    new_learning_set.append(np.array([new_matrix_a]))
    new_learning_set.append(np.array([new_matrix_b]))
    return nm_classification(training_set_after_sel_alg, new_learning_set, matrices_list, training_set)


def knm_evaluate_sets_first_step(learning_set, mean_list):
    distances = []
    for mean in mean_list:
        distances.append(calculate_distances_knm(mean, learning_set))

    print(distances[0])
    print(distances[1])
    #first time
    temp_a = []
    temp_b = []
    for i in range (len(learning_set)):
        if distances[0][i] < distances[1][i]:
            temp_a.append(learning_set[i])
        elif distances[1][i] < distances[0][i]:
            temp_b.append(learning_set[i])
        else:
            print("weszlo")
            temp_a.append(learning_set[i])
            temp_b.append(learning_set[i])

    print("Learning set [0]:", learning_set[0])
    print("temp_a:", temp_a)
    print("temp_b:", temp_b)
    mean_list[0] = calculate_mean(np.array([temp_a]), axis=1)
    mean_list[1] = calculate_mean(np.array([temp_b]), axis=1)
    print(mean_list[0])
    return temp_a, temp_b, mean_list


def knm_evaluate_sets_second_step(matrix_a, matrix_b, mean_list, learning_set):
    distances = []
    for mean in mean_list:
        distances.append(calculate_distances_knm(mean, learning_set))

    temp_a = []
    temp_b = []
    for i in range(len(learning_set)):
        if distances[0][i] < distances[1][i]:
            temp_a.append(learning_set[i])
        elif distances[1][i] < distances[0][i]:
            temp_b.append(learning_set[i])
        else:
            print("weszlo2")
            temp_a.append(learning_set[i])
            temp_b.append(learning_set[i])

    print("temp_a2:", temp_a)
    print("temp_b2:", temp_b)
    mean_list[0] = calculate_mean(np.array([temp_a]), axis=1)
    mean_list[1] = calculate_mean(np.array([temp_b]), axis=1)
    #if list(temp_a) == list(matrix_a):
    #    print("Gitara")

    if not (np.array_equal(temp_a, matrix_a) and np.array_equal(temp_b, matrix_b)):
        print("Gitara")
        new_matrix_a, new_matrix_b = knm_evaluate_sets_second_step(temp_a, temp_b, mean_list, learning_set)
    else:
        return temp_a, temp_b
    return new_matrix_a, new_matrix_b