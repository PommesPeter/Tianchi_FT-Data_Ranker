import unittest

from datasets import Dataset

from data_juicer.ops.filter.specified_field_lang_id_score_filter import \
    SpecifiedFieldLanguageIDScoreFilter
from data_juicer.utils.constant import Fields


class SpecifiedFieldLangIDScoreFilterTest(unittest.TestCase):

    def _run_language_id_score_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['instrution'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_en_case(self):

        ds_list = [{
            'instrution': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know",
            'instrution': '좀 더 유니크한 키워드로 다시 찾아줘'
        }, {
            'instrution': 'a v s e e f g a qkc'
        }, {
            'instrution': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?',
            'instrution': '请阅读一下文章'
        }, {
            'instrution': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = SpecifiedFieldLanguageIDScoreFilter(lang='en',
                                                 min_score=0.8,
                                                 text_key='instrution')
        self._run_language_id_score_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
