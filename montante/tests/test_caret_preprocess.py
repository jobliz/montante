import unittest

from montante.tests.BaseTest import BaseTest
from montante.examples import r_iris_raw
from montante.operations.R.functions import (r_c,
                                             r_summary,
                                             r_stats_predict,
                                             r_summary_to_pd,
                                             r_caret_preprocess,
                                             r_dataframe_subset_one_element,
                                             r_dataframe_column_index_from_name)


class TestCaretPreprocess(BaseTest):
    """
    Done following this post's output:

    https://machinelearningmastery.com/pre-process-your-dataset-in-r/
    """

    def setUp(self):
        super()
        self.iris_rdf = r_iris_raw()

    def _check_original_sepal_length(self, summary_df):
        self.assertEqual(summary_df['Sepal.Length'][0], 'Min.   :4.300  ')
        self.assertEqual(summary_df['Sepal.Length'][1], '1st Qu.:5.100  ')
        self.assertEqual(summary_df['Sepal.Length'][2], 'Median :5.800  ')
        self.assertEqual(summary_df['Sepal.Length'][3], 'Mean   :5.843  ')
        self.assertEqual(summary_df['Sepal.Length'][4], '3rd Qu.:6.400  ')
        self.assertEqual(summary_df['Sepal.Length'][5], 'Max.   :7.900  ')

    def _check_original_sepal_width(self, summary_df):
        self.assertEqual(summary_df['Sepal.Width'][0], 'Min.   :2.000  ')
        self.assertEqual(summary_df['Sepal.Width'][1], '1st Qu.:2.800  ')
        self.assertEqual(summary_df['Sepal.Width'][2], 'Median :3.000  ')
        self.assertEqual(summary_df['Sepal.Width'][3], 'Mean   :3.057  ')
        self.assertEqual(summary_df['Sepal.Width'][4], '3rd Qu.:3.300  ')
        self.assertEqual(summary_df['Sepal.Width'][5], 'Max.   :4.400  ')

    def _check_original_petal_length(self, summary_df):
        self.assertEqual(summary_df['Petal.Length'][0], 'Min.   :1.000  ')
        self.assertEqual(summary_df['Petal.Length'][1], '1st Qu.:1.600  ')
        self.assertEqual(summary_df['Petal.Length'][2], 'Median :4.350  ')
        self.assertEqual(summary_df['Petal.Length'][3], 'Mean   :3.758  ')
        self.assertEqual(summary_df['Petal.Length'][4], '3rd Qu.:5.100  ')
        self.assertEqual(summary_df['Petal.Length'][5], 'Max.   :6.900  ')

    def _check_original_petal_width(self, summary_df):
        self.assertEqual(summary_df['Petal.Width'][0], 'Min.   :0.100  ')
        self.assertEqual(summary_df['Petal.Width'][1], '1st Qu.:0.300  ')
        self.assertEqual(summary_df['Petal.Width'][2], 'Median :1.300  ')
        self.assertEqual(summary_df['Petal.Width'][3], 'Mean   :1.199  ')
        self.assertEqual(summary_df['Petal.Width'][4], '3rd Qu.:1.800  ')
        self.assertEqual(summary_df['Petal.Width'][5], 'Max.   :2.500  ')

    def test_original_summary(self):
        summary = r_summary(self.iris_rdf)
        summary_df = r_summary_to_pd(summary)

        self._check_original_sepal_length(summary_df)
        self._check_original_sepal_width(summary_df)
        self._check_original_petal_length(summary_df)
        self._check_original_petal_width(summary_df)

    def test_sepal_length_manual_center_scale(self):
        element_index = r_dataframe_column_index_from_name(self.iris_rdf, 'Sepal.Length')
        subset = r_dataframe_subset_one_element(self.iris_rdf, element_index)
        params = r_caret_preprocess(subset, method=r_c("center", "scale"))
        transformed = r_stats_predict(params, self.iris_rdf)
        summary = r_summary(transformed)
        summary_df = r_summary_to_pd(summary)

        # check that variables other that Sepal.Length haven't changed
        self._check_original_sepal_width(summary_df)
        self._check_original_petal_length(summary_df)
        self._check_original_petal_width(summary_df)

        # check expected changes
        self.assertEqual(summary_df['Sepal.Length'][0], 'Min.   :-1.86378  ')
        self.assertEqual(summary_df['Sepal.Length'][1], '1st Qu.:-0.89767  ')
        self.assertEqual(summary_df['Sepal.Length'][2], 'Median :-0.05233  ')
        self.assertEqual(summary_df['Sepal.Length'][3], 'Mean   : 0.00000  ')
        self.assertEqual(summary_df['Sepal.Length'][4], '3rd Qu.: 0.67225  ')
        self.assertEqual(summary_df['Sepal.Length'][5], 'Max.   : 2.48370  ')


if __name__ == '__main__':
    unittest.main()
