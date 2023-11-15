import unittest

from data_juicer.ops.mapper.remove_specific_lang_mapper import \
    RemoveSpecificLanguagesMapper


class RemoveSpecificLanguagesMapperTest(unittest.TestCase):

    def setUp(self):
        self.korean_op = RemoveSpecificLanguagesMapper(
            langs_to_remove='Korean')
        self.japanese_op = RemoveSpecificLanguagesMapper(
            langs_to_remove='Japanese')
        self.chinese_op = RemoveSpecificLanguagesMapper(
            langs_to_remove='Chinese')

    def _run_korean_helper(self, samples):
        for sample in samples:
            result = self.korean_op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def _run_chinese_helper(self, samples):
        for sample in samples:
            result = self.chinese_op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def _run_japanese_helper(self, samples):
        for sample in samples:
            result = self.japanese_op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_korean_text(self):

        samples = [{
            'text': 'continue\n 지금 번역하기',
            'target': 'continue\n Translate now'
        }, {
            'text':
            'great. Author, please indicate the final completed puzzle options \
                    in tabular form. Each row displays the question number, \
                    location, situation, and puzzle type. Please explain in \
                    detail.\n\nPlease write in formal tone, expository \
                    writing style, English language.\n \n \n \n 지금 번역하기',
            'target':
            'great. Author, please indicate the final completed puzzle options \
                    in tabular form. Each row displays the question number, \
                    location, situation, and puzzle type. Please explain in \
                    detail.\n\nPlease write in formal tone, expository \
                    writing style, English language.\n \n \n \n Translate now'
        }]

        self._run_korean_helper(samples)

    def test_chinese_text(self):

        samples = [{
            'text': 'continue\n 我是一个中国人',
            'target': 'continue\n I\'m Chinese.'
        }]

        self._run_chinese_helper(samples)

    def test_japanese_text(self):

        samples = [{
            'text':
            'continue\n この文章をCEFR B1相当の750～800単語の文章に書き換えてください',
            'target':
            'continue\n Please rewrite this text to 750 to 800 words equivalent\
             to CEFR B1'
        }, {
            'text': 'continue\n',
            'target': 'continue\n'
        }]

        self._run_japanese_helper(samples)


if __name__ == '__main__':
    unittest.main()
