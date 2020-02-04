import random
from src.selection import *
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


def main():
    path_to_file = "../Maple_Oak.txt"
    percentage = 20
    number_of_characteristics = 3
    selection = "SFS"  # SFS or F
    matrices = load_data(path_to_file)
    matrices_list = list(matrices.values())
    learning_set, training_set = divide_set_and_transpose(matrices_list, percentage)

    if selection == "F":
        print(fisher_algorithm(learning_set, number_of_characteristics))
    elif selection == "SFS":
        print(sfs_algorithm(learning_set, number_of_characteristics))


if __name__ == "__main__":
    main()