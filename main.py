import numpy as np
import random
from itertools import combinations


def load_data(path_to_file):
    matrices = {}
    with open(path_to_file) as file:
        all_lines = file.readlines()
        all_relevant_lines = all_lines[1::]
        for line in all_relevant_lines:
            line = line.replace('\n', '').split(',')
            name = line[0]
            numbers = line[1:]
            for i in range(len(numbers)):
                numbers[i] = float(numbers[i])
            name = name.split()[0]
            if name in matrices:
                matrices[name] = np.append(matrices.get(name), [numbers], axis=0)
            else:
                matrices[name] = np.array([numbers])
        file.close()
    return matrices


def divide_set_and_transpose(matrices, percentage):
    learning_set = []
    training_set = []
    for matrix in matrices:
        learning_set_temp = matrix
        training_set_temp = None
        how_many_to_training = (int)(len(matrix) * percentage * 0.01)

        for i in range(0, how_many_to_training):
            random_index = random.randint(0, len(learning_set_temp) - 1)
            if training_set_temp is None:
                training_set_temp = np.array([learning_set_temp[random_index]])
            else:
                training_set_temp = np.append(training_set_temp, [learning_set_temp[random_index]], axis=0)
            learning_set_temp = np.delete(learning_set_temp, random_index, axis=0)

        learning_set.append(np.transpose(learning_set_temp))
        training_set.append(np.transpose(training_set_temp))

    return learning_set, training_set


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

    mean_A = calculate_mean(learning_set[0], 1)
    mean_B = calculate_mean(learning_set[1], 1)
    all_combinations = calculate_combinations(len(learning_set[1]), number_of_characteristics)
    print(all_combinations)
    print(all_combinations[0][0])

    the_best_result = 0
    the_best_coordinates = None
    for coordinates in all_combinations:
        temp_mean_vector1 = []
        temp_mean_vector2 = []
        temp_matrix1 = []
        temp_matrix2 = []
        for i in coordinates:
            temp_mean_vector1.append(mean_A[i])
            temp_mean_vector2.append(mean_B[i])
            temp_matrix1.append(learning_set[0][i])
            temp_matrix2.append(learning_set[1][i])
        # distance between matrix
        numerator = np.linalg.norm(np.array(temp_mean_vector1) - (np.array(temp_mean_vector2)))
        # sum of standard deviation
        denominator = 0
        if len(coordinates) > 1:
            denominator = calculate_det(np.array(temp_matrix1)) + calculate_det(np.array(temp_matrix2))
        else:
            denominator = np.array(temp_matrix1).std() + np.array(temp_matrix2).std()
        f_results[coordinates] = (numerator / denominator)

        if the_best_result < f_results[coordinates]:
            the_best_result = f_results[coordinates]
            the_best_coordinates = coordinates

    # print("The best coordinates (f): " + str(the_best_coordinates))
    return the_best_coordinates

def main():
    file_name = "Maple_Oak.txt"
    percentage = 20
    number_of_characteristics = 2
    matrices = load_data(file_name)
    matrices_list = list(matrices.values())
    learning_set, training_set = divide_set_and_transpose(matrices_list, percentage)
    print(fisher_algorithm(learning_set, number_of_characteristics))


if __name__ == "__main__":
    main()