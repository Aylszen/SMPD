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


if __name__ == "__main__":
    unittest.main()