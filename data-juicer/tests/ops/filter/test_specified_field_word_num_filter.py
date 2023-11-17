import unittest

from datasets import Dataset

from data_juicer.ops.filter.specified_field_word_num_filter import \
    SpecifiedFieldWordNumFilter
from data_juicer.utils.constant import Fields


class SpecifiedFieldWordNumFilterTest(unittest.TestCase):

    def _run_word_num_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['output'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'output': 'Today is Sun'
        }, {
            'output':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'output': 'a v s e c s f e f g a a a  '
        }, {
            'output': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'output': '...'
        }]
        tgt_list = [{
            'output':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'output': 'a v s e c s f e f g a a a  '
        }]
        dataset = Dataset.from_list(ds_list)
        op = SpecifiedFieldWordNumFilter(min_num=5,
                                         max_num=15,
                                         text_key='output')
        self._run_word_num_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
