import logging
import unittest

from nipype.algorithms.icc import ICC_rep_anova
# from nltools.data.brain_data import Brain_Data
import numpy as np

# from helpers.icc import compute_ICC
# from helpers.icc import compute_icc_scores_pingouin
from helpers.i2c2 import compute_i2c2

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


def compute_ICC_1_1_manually(observations: np.array):
    Y = observations

    n, reps = np.shape(Y)
    grand_mean = np.mean(Y)

    sub_means = np.atleast_2d(np.mean(Y, axis=1)).T
    within_per_sub = np.sum((sub_means - Y) ** 2, axis=1) / (reps - 1)
    within_sub = np.sum(within_per_sub) / (n - 1)

    between_sub = np.sum((sub_means - grand_mean) ** 2) / (n - 1)

    ICC = within_sub / (between_sub + within_sub)
    return ICC


class TestTestRetest(unittest.TestCase):

    # Define test data sets.

    def _get_Y(self):
        # see table 2 in P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass
        # Correlations: Uses in Assessing Rater Reliability". Psychological
        # Bulletin 86 (2): 420-428
        y = np.array(
            [
                [9, 2, 5, 8],
                [6, 1, 3, 2],
                [8, 4, 6, 8],
                [7, 1, 2, 6],
                [10, 5, 6, 9],
                [6, 2, 4, 7]
            ]
        )
        print('y:', y.shape)
        return y

    def _get_Y_multiples(self):
        y_multiples = np.array(
            [
                [
                    [9, 2, 5, 8],
                    [6, 1, 3, 2],
                    [8, 4, 6, 8]
                ],
                [
                    [4, 3, 4, 9],
                    [5, 2, 2, 1],
                    [9, 5, 7, 7]
                ]
            ]
        )
        y_multiples = np.transpose(y_multiples, (1, 2, 0))
        print('y:', y_multiples.shape)
        return y_multiples

    # Test ICC computations.

    # def test_pingouin_ICC_1_1(self):
    #     icc_1_1_score = compute_ICC_pingouin(observations=self._get_Y(), icc_type='ICC1')
    #     self.assertAlmostEqual(icc_1_1_score, 0.17, 2)

    # def test_pingouin_ICC_2_1(self):
    #     icc_2_1_score = compute_ICC_pingouin(observations=self._get_Y(), icc_type='ICC2')
    #     self.assertAlmostEqual(icc_2_1_score, 0.29, 2)

    # def test_pingouin_ICC_3_1(self):
    #     icc_3_1_score = compute_ICC_pingouin(observations=self._get_Y(), icc_type='ICC3')
    #     self.assertAlmostEqual(icc_3_1_score, 0.71, 2)

    # def test_manual_ICC_1_1(self):
    #     y = self._get_Y()
    #     icc_1_1 = compute_ICC_1_1_manually(observations=y)
    #     self.assertEqual(round(icc_1_1, 2), 0.71)

    # def test_nltools_ICC_2_1(self):
    #     y = self._get_Y()
    #     icc_2_1 = compute_ICC(observations=y, icc_type="icc2")
    #     self.assertEqual(round(icc_2_1, 2), 0.71)

    # def test_nltools_ICC_3_1(self):
    #     y = self._get_Y()
    #     icc_3_1 = compute_ICC(observations=y, icc_type="icc3")
    #     self.assertEqual(round(icc_3_1, 2), 0.71)

    def test_ICC_rep_anova(self):
        """
        This test function is taken from NiPyPe directly
        It is ICC(3, 1)
        :return:
        """
        Y = self._get_Y()
        icc, r_var, e_var, _, dfc, dfe = ICC_rep_anova(Y)
        # see table 4
        assert round(icc, 2) == 0.71
        assert dfc == 3
        assert dfe == 15
        assert np.isclose(r_var / (r_var + e_var), icc)

    def manual_ICC_compute(self, observations: np.array):
        """
        ICC computation definition is taken from Choe2017.
        Their definition of between-subject and within-session is not given.
        :param observations: an array of shape (n_subjects, n_repeated_measures).
        :return:
        """
        average_session_score = np.mean(observations, axis=0)
        within_subject_variance = np.var(average_session_score)
        print('within subject variance:', within_subject_variance)

        average_subject_score = np.mean(observations, axis=1)
        between_subject_variance = np.var(average_subject_score)
        print('between subject variance:', between_subject_variance)

        icc = between_subject_variance / (between_subject_variance + within_subject_variance)
        print('icc score (manual):', icc)
        return icc

    # def test_manual_ICC_compute(self):
    #     Y = self._get_Y()
    #     icc = self.manual_ICC_compute(Y)
    #
    #     self.assertEqual(round(icc, 2), 0.71)
    #     # assert round(icc, 2) == 0.71

    # Test I2C2 computations.

    def test_compute_I2C2(self):
        # Y = self._get_Y_multiples()

        y = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [23, 23, 23],
            [23, 23, 23],
            [23, 23, 23],
            [23, 23, 23],
            [13, 13, 13],
            [13, 13, 13],
            [13, 13, 13],
            [13, 13, 13],
            [7, 7, 7],
            [7, 7, 7],
            [7, 7, 7],
            [7, 7, 7],
            [3, 4, 3],
            [3, 4, 3],
            [3, 4, 3],
            [3, 4, 3],
            [9, 9, 9],
            [9, 9, 9],
            [9, 9, 9],
            [9, 9, 9],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ])
        # y = np.random.random(size=(n, p))
        print('y:', y.shape)

        n_subjects = 7
        n_visits = 4

        D = 3  # e.g. number of ICA components

        n_voxels = int(D * (D - 1) / 2)
        print('n_voxels:', n_voxels)

        p = n_voxels
        n = n_subjects * n_visits

        i2c2_score = compute_i2c2(
            y=y,
            n_subjects=n_subjects,
            n_scans=n_visits
        )
        self.assertAlmostEqual(i2c2_score, 0.71, 2)


if __name__ == "__main__":
    unittest.main()
