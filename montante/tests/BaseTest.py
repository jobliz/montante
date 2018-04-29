import unittest

import pandas as pd

from montante.examples import data_example_loader


class BaseTest(unittest.TestCase):

    def setUp(self):
        super()

    def _iris_dataset(self) -> pd.DataFrame:
        return data_example_loader('sklearn_iris')[0]

    def _iris_payload(self):
        return {
            'engine': 'caret',
            'target': 'target',
            'predictors': ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm'],
            'engine-parameters': {
                'method': 'C5.0',
                'preprocess': [],
                'metric': 'Accuracy',
                'training-control': {
                    'method': 'boot',
                    'number': 5,
                    'repeats': 1
                }
            }
        }
