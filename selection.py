import numpy as np
from itertools import combinations


def calculate_mean(matrix, axis):
    return matrix.mean(axis)


def calculate_combinations(matrix_length, dimension):
    rows = []
    for i in range(matrix_length):
        rows.append(i)
    return list(combinations(rows, dimension))


def calculate_det(matrix):
    mean = matrix.mean(1)
    for row in range(np.size(matrix, 0)):
        for column in range(np.size(matrix, 1)):
            matrix[row][column] = matrix[row][column] - mean[row]
    matrix_std = 1 / np.size(matrix, 1) * np.matmul(matrix, matrix.transpose())
    return np.linalg.det(matrix_std)


def fisher_algorithm(learning_set, number_of_characteristics):
    f_results = {}

    mean_a = calculate_mean(learning_set[0], 1)
    mean_b = calculate_mean(learning_set[1], 1)
    all_combinations = calculate_combinations(len(learning_set[1]), number_of_characteristics)

    the_best_result = 0
    the_best_coordinates = None
    for coordinates in all_combinations:
        temp_mean_vector1 = []
        temp_mean_vector2 = []
        temp_matrix1 = []
        temp_matrix2 = []
        for i in coordinates:
            temp_mean_vector1.append(mean_a[i])
            temp_mean_vector2.append(mean_b[i])
            temp_matrix1.append(learning_set[0][i])
            temp_matrix2.append(learning_set[1][i])

        mean_subtraction = np.linalg.norm(np.array(temp_mean_vector1) - (np.array(temp_mean_vector2)))

        sum_of_dets = 0
        if len(coordinates) > 1:
            sum_of_dets = calculate_det(np.array(temp_matrix1)) + calculate_det(np.array(temp_matrix2))
        else:
            sum_of_dets = np.array(temp_matrix1).std() + np.array(temp_matrix2).std()

        f_results[coordinates] = (mean_subtraction / sum_of_dets)

        if the_best_result < f_results[coordinates]:
            the_best_result = f_results[coordinates]
            the_best_coordinates = coordinates

    return the_best_coordinates

