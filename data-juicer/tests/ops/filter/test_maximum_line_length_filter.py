import unittest

from datasets import Dataset

from data_juicer.ops.filter.maximum_line_length_filter import \
    MaximumLineLengthFilter
from data_juicer.utils.constant import Fields


class MaximumLineLengthFilterTest(unittest.TestCase):

    def _run_maximum_line_length_filter(self, dataset: Dataset, target_list,
                                        op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = [{
            'text': 'a v s e e f g a qkc'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        dataset = Dataset.from_list(ds_list)
        op = MaximumLineLengthFilter(min_len=10, max_len=20)
        self._run_maximum_line_length_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
