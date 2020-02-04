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
    mean_a = calculate_mean(learning_set[0], 1)
    mean_b = calculate_mean(learning_set[1], 1)
    all_char_combinations = calculate_combinations(len(learning_set[1]), number_of_characteristics)

    the_best_result = 0
    the_best_characteristics = None
    for char_combination in all_char_combinations:
        temp_mean_a = []
        temp_mean_b = []
        temp_matrix_a = []
        temp_matrix_b = []
        for i in char_combination:
            temp_mean_a.append(mean_a[i])
            temp_mean_b.append(mean_b[i])
            temp_matrix_a.append(learning_set[0][i])
            temp_matrix_b.append(learning_set[1][i])

        mean_subtraction = np.linalg.norm(np.array(temp_mean_a) - (np.array(temp_mean_b)))

        if len(char_combination) > 1:
            sum_of_matrices = calculate_det(np.array(temp_matrix_a)) + calculate_det(np.array(temp_matrix_b))
        else:
            sum_of_matrices = np.array(temp_matrix_a).std() + np.array(temp_matrix_b).std()

        f_results = (mean_subtraction / sum_of_matrices)

        if the_best_result < f_results:
            the_best_result = f_results
            the_best_characteristics = char_combination

    return the_best_characteristics


def sfs_algorithm(learning_set, number_of_characteristics):
    f_results = {}

    mean_a = calculate_mean(learning_set[0], 1)
    mean_b = calculate_mean(learning_set[1], 1)

    the_best_diff = 0
    the_best_coordinates = ()
    for d in range(1, number_of_characteristics + 1):
        # refresh value
        the_best_diff = 0
        # print(d)
        # get all possible combinations
        all_combinations = calculate_combinations(len(learning_set[0]), d)
        chosen_combinations = []
        
        if the_best_coordinates:
            for coordinates_candidate in all_combinations:
                flag = True
                for i in the_best_coordinates:
                    flag = i in coordinates_candidate
                    if not flag:
                        break
                if flag:
                    chosen_combinations.append(coordinates_candidate)
        # print("Chosen comb:" + str(chosen_combinations))
        if not chosen_combinations:
            chosen_combinations = all_combinations

        for coordinates in chosen_combinations:
            temp_mean_vector1 = []
            temp_mean_vector2 = []
            temp_matrix1 = []
            temp_matrix2 = []
            for i in coordinates:
                temp_mean_vector1.append(mean_a[i])
                temp_mean_vector2.append(mean_b[i])
                temp_matrix1.append(learning_set[0][i])
                temp_matrix2.append(learning_set[1][i])
            # distance between matrix
            mean_subtraction = np.linalg.norm(np.array(temp_mean_vector1) - (np.array(temp_mean_vector2)))

            denominator = 0
            if len(coordinates) > 1:
                denominator = calculate_det(np.array(temp_matrix1)) + calculate_det(np.array(temp_matrix2))
            else:
                denominator = np.array(temp_matrix1).std() + np.array(temp_matrix2).std()

            f_results[coordinates] = (mean_subtraction / denominator)

            if the_best_diff < f_results[coordinates]:
                the_best_diff = f_results[coordinates]
                the_best_coordinates = coordinates

    return the_best_coordinates
