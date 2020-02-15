from src.classification import *
import unittest


class TestClassification(unittest.TestCase):

    def test_calculate_distances(self):
        pattern = np.array([[1, 7, 3]])
        matrix = np.array([[1, 2, 2, 4], [2, 5, 5, 7], [3, 3, 4, 5]]).transpose()
        expected_results = [2.23606797749979, 2.449489742783178, 3.605551275463989, 5.0]
        result = calculate_distances(pattern, matrix)
        self.assertEqual(expected_results, result)

    def test_calculate_distances(self):
        pattern = np.array([[1, 7, 3]])
        matrix = np.array([[1, 2, 2, 4], [2, 5, 5, 7], [3, 3, 4, 5]]).transpose()
        expected_results = [2.23606797749979, 2.449489742783178, 3.605551275463989, 5.0]
        result = calculate_distances(pattern, matrix)
        self.assertEqual(expected_results, result)

    def test_calculate_mean(self):
        matrix_a = np.array([[1, 2, 2, 4], [2, 5, 5, 7], [3, 3, 4, 5]]).transpose()
        expected_results = np.array([2.25, 4.75, 3.75])
        result = calculate_mean(matrix_a, 0)
        self.assertEqual(expected_results.all(), result.all())


if __name__ == "__main__":
    unittest.main()