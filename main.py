import numpy as np

A = [[3, 2, 2, 1]]
print(A)
new_array_2 = np.vstack([A , [[2, 3, 0, 7]]])
print(new_array_2)
new_array = np.vstack([new_array_2, [[-1, 2, 2,-3]]])
print(new_array)


def matrix_mean(matrix, axis_value):
    return np.mean(matrix, axis=axis_value)


matrix_mean2 = matrix_mean(new_array, 1)
print (matrix_mean2)

def add_row_to_array(old_array, row_data):
    new_array = np.vstack([old_array, [row_data]])
    return new_array


def euclidean_distance(matrix, mean_matrix):
    print(mean_matrix)
    for element in mean_matrix:
        print(element)
    i = 0
    k = 0
    for column in matrix.T:
        for element in column:
            k += np.sum((element - mean_matrix[i]) ** 2)
            print("Element")
            print(element)
            print ("mean_matrix")
            mean_matrix[i]
            print(np.sum((element - mean_matrix[i]) ** 2))
            i+=1
        i=0
        print("Weszlo")
        #np.sqrt(np.sum((v - u) ** 2))
    print(np.sqrt(k))

print("###")
print(euclidean_distance(new_array, matrix_mean2))

#temp_new_array = add_row_to_array(new_array, [9,10,2])
#print(temp_new_array)


def fisher_algorithm(data):
    print("Hello")