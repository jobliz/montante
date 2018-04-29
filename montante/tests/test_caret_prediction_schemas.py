from jsonschema import Draft4Validator

from montante.tests.BaseTest import BaseTest
from montante.schemas.train import create_specific_training_schema
from montante.util import use_validator


class CaretTrainingSchemaTests(BaseTest):
    """
    Tests for the schemas intended to train caret prediction models.
    """

    def _create_payload(self):
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

    def test_caret_training_error_unexisting_method_in_function_call(self):
        # TODO: Custom errors, to tell apart engine not implemented from method not existent.
        with self.assertRaises(NotImplementedError):
            create_specific_training_schema('caret', 'UNEXISTENT')

    def test_caret_training_error_no_engine_parameters(self):
        payload = self._create_payload()
        del(payload['engine-parameters'])
        schema = create_specific_training_schema('caret', 'C5.0')
        errors = use_validator(Draft4Validator(schema), payload)
        # TODO: specific 'engine-parameters' element, not a message string
        self.assertEqual(errors[0], (['required'], "'engine-parameters' is a required property"))

    def test_caret_training_error_unexisting_method(self):
        payload = self._create_payload()
        payload['engine-parameters']['method'] = 'UNEXISTENT'
        schema = create_specific_training_schema('caret', 'C5.0')
        errors = use_validator(Draft4Validator(schema), payload)
        self.assertEqual(errors[0][0], ['properties', 'engine-parameters', 'properties', 'method', 'enum'])

    def test_caret_training_error_unexisting_training_method(self):
        payload = self._create_payload()
        payload['engine-parameters']['training-control']['method'] = 'UNEXISTENT'
        schema = create_specific_training_schema('caret', 'C5.0')
        errors = use_validator(Draft4Validator(schema), payload)
        expected = ['properties', 'engine-parameters', 'properties', 'training-control', 'properties', 'method', 'enum']
        self.assertEqual(errors[0][0], expected)
