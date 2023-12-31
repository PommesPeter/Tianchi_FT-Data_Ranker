import unittest

from data_juicer.ops.mapper.keyword_mapper import KeywordMapper


class KeywordMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = KeywordMapper()

    def _run_helper(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_only_list_text(self):

        samples = [{
            'text': 'continue\n지금 번역하기',
            'target': 'continue\n'
        }, {
            'text':
            'great. Author, please indicate the final completed puzzle\
                 options in tabular form. Each row displays the question\
                 number, location, situation, and puzzle type. Please explain\
                 in detail.\n\nPlease write in formal tone, expository writing\
                 style, English language.\n \n \n \n 지금 번역하기',
            'target':
            'great. Author, please indicate the final completed puzzle\
                 options in tabular form. Each row displays the question \
                 number, location, situation, and puzzle type. Please explain\
                 in detail.\n\nPlease write in formal tone, expository writing\
                 style, English language.\n \n \n \n '
        }]

        self._run_helper(samples)


if __name__ == '__main__':
    unittest.main()
