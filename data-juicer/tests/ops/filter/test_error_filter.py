import unittest

from datasets import Dataset

from data_juicer.ops.filter.error_filter import ErrorFilter


class ErrorFilterTest(unittest.TestCase):

    def _run_suffix_filter(self, dataset: Dataset, target_list, op):
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text':
            '!, network error, There was an error generating a response',
        }, {
            'text': 'a v s e c s f e f g a a a  ',
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
        }, {
            'text':
            'dasdasdasdasdasdasdasd, There was an error generating a response',
        }]
        tgt_list = [{
            'text': 'a v s e c s f e f g a a a  ',
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
        }]
        dataset = Dataset.from_list(ds_list)
        op = ErrorFilter(errors=[
            'network error', 'Network error',
            'There was an error generating a response'
        ])
        self._run_suffix_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
