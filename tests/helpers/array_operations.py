import unittest

from nilearn import connectome
import numpy as np

from helpers.array_operations import _check_symmetric
from helpers.array_operations import map_array, reorder_symmetric_matrix
from helpers.array_operations import reconstruct_symmetric_matrix_from_tril
from helpers.array_operations import slice_covariance_structure


class TestArrayOperations(unittest.TestCase):

    def test_slice_covariance_structure(self):
        test_full_covariance_structure = self._get_test_correlation_structure()
        test_edge_indices = (1, 0)
        test_sliced_covariance_structure = slice_covariance_structure(
            test_full_covariance_structure, test_edge_indices
        )
        true_sliced_covariance_structure = np.array([
            [[1.0, 0.3],
             [0.3, 1.0]],
            [[1.0, 0.1],
             [0.1, 1.0]]
        ])
        np.testing.assert_array_equal(test_sliced_covariance_structure, true_sliced_covariance_structure)

    @staticmethod
    def _get_test_array() -> np.array:
        return np.arange(4)

    @staticmethod
    def _get_test_array_map() -> dict:
        return {
            0: 3,
            1: 0,
            2: 2,
            3: 1
        }

    def test_map_array(self):
        orig_array = self._get_test_array()
        array_map = self._get_test_array_map()
        mapped_array = map_array(orig_array, array_map)
        np.testing.assert_array_equal(mapped_array, np.array([3, 0, 2, 1]))

    @staticmethod
    def _get_test_symmetric_matrix() -> np.array:
        return np.array([
            [1., 2., 3., 4., 5.],
            [2., 1., 6., 7., 8.],
            [3., 6., 1., 9., 1.],
            [4., 7., 9., 1., 2.],
            [5., 8., 1., 2., 1.]
        ])

    def test_reorder_symmetric_matrix(self):
        test_symmetric_matrix = self._get_test_symmetric_matrix()
        new_index = [1, 0, 2, 4, 3]
        reordered_symmetric_matrix = reorder_symmetric_matrix(
            original_matrix=test_symmetric_matrix,
            new_order=new_index
        )
        np.testing.assert_array_equal(
            reordered_symmetric_matrix,
            np.array([
                [1., 2., 6., 8., 7.],
                [2., 1., 3., 5., 4.],
                [6., 3., 1., 1., 9.],
                [8., 5., 1., 1., 2.],
                [7., 4., 9., 2., 1.],
            ])
        )

    @staticmethod
    def _get_test_correlation_matrix() -> np.array:
        return np.array([
            [1.0, 0.3, 0.5],
            [0.3, 1.0, 0.2],
            [0.5, 0.2, 1.0]
        ])

    def test_reconstruct_correlation_matrix_from_tril(self):
        test_corr_matrix = self._get_test_correlation_matrix()  # (D, D)
        n_time_series = test_corr_matrix.shape[0]
        test_corr_matrix_tril = connectome.sym_matrix_to_vec(test_corr_matrix, discard_diagonal=True)  # (D*(D-1)/2, )
        reconstructed_test_corr_matrix = reconstruct_symmetric_matrix_from_tril(
            test_corr_matrix_tril,
            n_time_series=n_time_series
        )  # (D, D)
        self.assertTrue(_check_symmetric(reconstructed_test_corr_matrix))
        np.testing.assert_array_equal(reconstructed_test_corr_matrix, test_corr_matrix)

    @staticmethod
    def _get_test_correlation_structure() -> np.array:
        return np.array([
            [[1.0, 0.3, 0.5],
             [0.3, 1.0, 0.2],
             [0.5, 0.2, 1.0]],
            [[1.0, 0.1, 0.4],
             [0.1, 1.0, 0.3],
             [0.4, 0.3, 1.0]]
        ])

    # def test_compute_rate_of_change(self):
    #     test_corr_structure = self._get_test_correlation_structure()  # (n_matrices, D, D)
    #     average_roc = compute_rate_of_change(test_corr_structure)  # (D, D)
    #     np.testing.assert_array_equal(average_roc, average_roc)


if __name__ == '__main__':
    unittest.main()
