import random
from src.selection import *
from src.classification import *
from copy import deepcopy


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
        learning_set_temp = deepcopy(matrix)
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


def create_matrices_based_on_sel_alg(the_best_characteristics, learning_set):
    new_matrix_a = None
    new_matrix_b = None

    for characteristic in the_best_characteristics:
        if new_matrix_a is None:
            new_matrix_a = np.array([learning_set[0][characteristic]])
            new_matrix_b = np.array([learning_set[1][characteristic]])
        else:
            new_matrix_a = np.append(new_matrix_a, [learning_set[0][characteristic]], axis=0)
            new_matrix_b = np.append(new_matrix_b, [learning_set[1][characteristic]], axis=0)


    new_matrix = []
    new_matrix.append(new_matrix_a)
    new_matrix.append(new_matrix_b)
    return new_matrix


def main():
    path_to_file = "../Maple_Oak.txt"
    percentage = 20
    number_of_characteristics = 3
    selection = "SFS"  # SFS or F
    classification = "NN"  # NN kNN
    matrices = load_data(path_to_file)
    matrices_list = list(matrices.values())
    learning_set, training_set = divide_set_and_transpose(matrices_list, percentage)

    if selection == "F":
        the_best_characteristics = fisher_algorithm(learning_set, number_of_characteristics)
    elif selection == "SFS":
        the_best_characteristics = sfs_algorithm(learning_set, number_of_characteristics)
    print("The best characteristics for", selection, ":", the_best_characteristics)

    learning_set_after_sel_alg = create_matrices_based_on_sel_alg(the_best_characteristics, learning_set)
    training_set_after_sel_alg = create_matrices_based_on_sel_alg(the_best_characteristics, training_set)

    if classification == "NN":
        indexes = nn_classification(training_set_after_sel_alg, learning_set_after_sel_alg)
    elif classification == "kNN":
        print("Not implemented yet")

    print(calculate_efficiency(indexes, matrices_list, training_set))


if __name__ == "__main__":
    main()
