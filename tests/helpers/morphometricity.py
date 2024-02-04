import os
import sys
import unittest

import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise_distances
from statsmodels.stats.correlation_tools import cov_nearest

from helpers.morphometricity import variance_component_model


class TestMorphometricity(unittest.TestCase):

    def test_variance_component_model(self):
        """
        Source:
            https://github.com/sina-mansour/neural-identity/blob/bf8855b3443737f5c660ea242c43282bf9d2071f/notebooks/VCM.ipynb
        """
        data = sns.load_dataset('iris')

        # The variable of interest (trait values) is `petal_length`.
        y = np.array(data['petal_length']).reshape(-1, 1)  # (n_samples, 1)

        # covariates: sepal_width and sepal_length
        X = np.array(data[['sepal_length', 'sepal_width']])  # (n_samples, n_covariates=2)

        # Anatomical similarity: computed as inverted euclidean distance in petal_width
        euclidean_dist = pairwise_distances(
            np.array(data['petal_width']).reshape(-1, 1),
            metric='euclidean'
        )
        K = cov_nearest(1 - (euclidean_dist / np.max(euclidean_dist)))  # (n_samples, n_samples)

        # fit variance component model and report the ratio of overall variance explained by the similarity metric
        result = variance_component_model(y, X, K)
        print(result)
        print('--- Variance ratio in petal_length explained by petal width: {:.2f}'.format(result['m2']))

        self.assertAlmostEqual(result['m2'], 0.91, 2)


if __name__ == '__main__':
    unittest.main()
