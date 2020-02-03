import numpy as np


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


def main():
    matrices = load_data("Maple_Oak.txt")
    print(matrices["Acer"])


if __name__ == "__main__":
    main()