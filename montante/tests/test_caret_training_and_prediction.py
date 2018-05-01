import unittest

from rpy2.rinterface import RRuntimeError

from montante.tests.BaseTest import BaseTest
from montante.operations.train import training_operation
from montante.operations.predict import prediction_operation


class TestCaretTrainingAndPrediction(BaseTest):

    def setUp(self):
        super()
        self.iris_df = self._iris_dataset()
        self.caret_c50 = training_operation(self.iris_df, self._iris_payload())

    def test_prediction_ok(self):
        p = prediction_operation(self.caret_c50, {
            'petal_width_cm': [1, 1, 1],
            'sepal_length_cm': [1, 1, 1],
            'sepal_width_cm': [1, 1, 1],
            'petal_length_cm': [1, 1, 1]
        })

        self.assertEqual(len(p), 3)

    def test_prediction_bad_type_whithout_checking(self):
        with self.assertRaises(RRuntimeError):
            prediction_operation(self.caret_c50, {
                'petal_width_cm': ["a", "b", "c"],
                'sepal_length_cm': [1, 1, 1],
                'sepal_width_cm': [1, 1, 1],
                'petal_length_cm': [1, 1, 1]
            })

    def test_prediction_with_column_missing(self):
        with self.assertRaises(RRuntimeError):
            prediction_operation(self.caret_c50, {
                'sepal_length_cm': [1, 1, 1],
                'sepal_width_cm': [1, 1, 1],
                'petal_length_cm': [1, 1, 1]
            })

    def test_prediction_with_unrecognized_extra_column(self):
        with self.assertRaises(RRuntimeError):
            prediction_operation(self.caret_c50, {
                'sepal_length_cm': [1, 1, 1],
                'sepal_width_cm': [1, 1, 1],
                'petal_length_cm': [1, 1, 1],
                'random_string': [1, 1, 1]
            })


if __name__ == '__main__':
    unittest.main()
