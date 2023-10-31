import unittest

from data_juicer.ops.mapper.clean_links_mapper import CleanLinksMapper


class CleanLinksMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = CleanLinksMapper()

    def _run_clean_links(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_lower_ftp_links_text(self):

        samples = [{
            'text': 'ftp://user:password@ftp.example.com:21/',
            'target': ''
        }, {
            'text': 'ftp://www.example.com/path/to/file.txt',
            'target': ''
        }, {
            'text': 'ftp://example.com/my-page.html',
            'target': ''
        }, {
            'text': 'ftp://example.com',
            'target': ''
        }]
        self._run_clean_links(samples)

    def test_upper_ftp_links_text(self):

        samples = [{
            'text': 'FTP://user:password@ftp.example.COm:21/',
            'target': ''
        }, {
            'text': 'FTP://www.example.com/path/to/file.txt',
            'target': ''
        }, {
            'text': 'Ftp://example.com/my-page.HTMl',
            'target': ''
        }, {
            'text': 'FTP://EXAMPLE.COM',
            'target': ''
        }]
        self._run_clean_links(samples)

    def test_lower_https_links_text(self):

        samples = [{
            'text':
            'https://www.example.com/file.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text':
            'https://example.com/my-page.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text': 'https://example.com',
            'target': ''
        }]
        self._run_clean_links(samples)

    def test_upper_https_links_text(self):

        samples = [{
            'text':
            'hTTps://www.example.com/file.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text':
            'HttpS://example.Com/my-page.HTML?param1=value1&param2=value2',
            'target': ''
        }, {
            'text': 'HTTPS://EXAMPLE.COM',
            'target': ''
        }]
        self._run_clean_links(samples)

    def test_mixed_https_links_text(self):

        samples = [{
            'text': 'This is a test,'
            'https://www.example.com/file.html?param1=value1&param2=value2',
            'target': 'This is a test,'
        }, {
            'text': '这是个测试,'
            'https://example.com/my-page.html?param1=value1&param2=value2',
            'target': '这是个测试,'
        }, {
            'text': '这是个测试,https://example.com',
            'target': '这是个测试,'
        }]
        self._run_clean_links(samples)

    def test_lower_http_links_text(self):

        samples = [{
            'text':
            'http://example.com/my-page.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text':
            'http://www.example.com/file.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text': 'https://example.com',
            'target': ''
        }]
        self._run_clean_links(samples)

    def test_upper_http_links_text(self):

        samples = [
            {
                'text':
                'HTTP://example.com/my-page.html?param1=value1&param2=value2',
                'target': ''
            },
            {
                'text':
                'hTTp://www.example.com/file.html?param1=value1&param2=value2',
                'target': ''
            },
            {
                'text': 'HTTPS://EXAMPLE.COM',
                'target': ''
            },
        ]
        self._run_clean_links(samples)

    def test_mixed_http_links_text(self):

        samples = [{
            'text': 'This is a test,'
            'http://www.example.com/file.html?param1=value1&param2=value2',
            'target': 'This is a test,'
        }, {
            'text': '这是个测试,'
            'http://example.com/my-page.html?param1=value1&param2=value2',
            'target': '这是个测试,'
        }, {
            'text': '这是个测试,https://example.com',
            'target': '这是个测试,'
        }]
        self._run_clean_links(samples)

    def test_email_text(self):

        samples = [
            {
                'text': 'This is a sample@example for test',
                'target': 'This is a sample@example for test',
            },
            {
                'text': '这是一个测试, sample@example',
                'target': '这是一个测试, sample@example',
            },
        ]
        self._run_clean_links(samples)

    def test_fake_links_text(self):

        samples = [
            {
                'text': 'abcd:/e f is a sample for test',
                'target': 'abcd:/e f is a sample for test',
            },
            {
                'text': 'abcd://ef is a sample for test',
                'target': ' is a sample for test',
            },
            {
                'text': 'This is a test,'
                'http测试://www.example.com/file.html?param1=value1',
                'target': 'This is a test,'
            },
            {
                'text': 'This is a test,'
                'http://www.测试.com/path/file.html?param1=value1&param2=value2',
                'target': 'This is a test,'
            },
        ]
        self._run_clean_links(samples)

    def test_no_link_text(self):

        samples = [
            {
                'text': 'This is a sample for test',
                'target': 'This is a sample for test',
            },
            {
                'text': '这是一个测试',
                'target': '这是一个测试',
            },
            {
                'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►',
                'target': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►',
            },
        ]
        self._run_clean_links(samples)


if __name__ == '__main__':
    unittest.main()
