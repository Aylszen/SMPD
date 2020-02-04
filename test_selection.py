from selection import *
import unittest


class TestSelection(unittest.TestCase):

    def test_mean_subtraction(self):
        mean_a = np.array([[2, 3]])
        mean_b = np.array([[3, 5]])
        expected_result = 2.23606797749979
        mean_result = np.linalg.norm(np.array(mean_a) - (np.array(mean_b)))
        self.assertEqual(expected_result, mean_result)

    def test_calculate_det(self):
        matrix = np.array([[3, 2, 2, 1], [2, 3, 0, 7]])
        expected_result = 1.6875000000000002
        det_result = calculate_det(matrix)
        self.assertEqual(expected_result, det_result)

    def test_fisher_algorithm_one_best_characteristic(self):
        matrices_list = []
        number_of_characteristics = 1
        matrix_a = np.array([[3, 2, 2, 1], [2, 3, 0, 7], [-1, 2, 2, -3]])
        matrix_b = np.array([[3, 1, 5], [6, 4, 5], [-1, 3, -2]])
        matrices_list.append(matrix_a)
        matrices_list.append(matrix_b)
        # Second characteristic (like in classes) = (1,)
        expected_result = (1,)
        fisher_result = fisher_algorithm(matrices_list, number_of_characteristics)
        self.assertEqual(expected_result, fisher_result)

    def test_fisher_algorithm_two_best_characteristics(self):
        matrices_list = []
        number_of_characteristics = 2
        matrix_a = np.array([[3, 2, 2, 1], [2, 3, 0, 7], [-1, 2, 2, -3]])
        matrix_b = np.array([[3, 1, 5], [6, 4, 5], [-1, 3, -2]])
        matrices_list.append(matrix_a)
        matrices_list.append(matrix_b)
        # First and second characteristic (like in classes) = (0, 1)
        expected_result = (0, 1)
        fisher_result = fisher_algorithm(matrices_list, number_of_characteristics)
        self.assertEqual(expected_result, fisher_result)


if __name__ == "__main__":
    unittest.main()