import unittest

from data_juicer.ops.mapper.clean_ip_mapper import CleanIpMapper


class CleanIpMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = CleanIpMapper()

    def _run_clean_ip(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_ipv4(self):

        samples = [{
            'text': 'test of ip 234.128.124.123',
            'target': 'test of ip '
        }, {
            'text': '34.0.124.123',
            'target': ''
        }, {
            'text': 'ftp://example.com/188.46.244.216my-page.html',
            'target': 'ftp://example.com/my-page.html'
        }, {
            'text': 'ft174.1421.237.246my',
            'target': 'ft174.1421.237.246my'
        }]
        self._run_clean_ip(samples)

    def test_ipv6(self):

        samples = [{
            'text': 'dd41:cbaf:d1b4:10a0:b215:72e3:6eaf:3ecb',
            'target': ''
        }, {
            'text': 'test of ip 4394:538a:3bf3:61c3:cb0d:d214:526f:70d',
            'target': 'test of ip '
        }, {
            'text': 'com/f770:c52e:ddce:3a9f:8c3b:a7bd:d81f:985cmy-page.html',
            'target': 'com/my-page.html'
        }, {
            'text': 'ft1926:43a1:fcb5:ees06:ae63:a2a4:c656:d014my',
            'target': 'ft1926:43a1:fcb5:ees06:ae63:a2a4:c656:d014my'
        }]
        self._run_clean_ip(samples)


if __name__ == '__main__':
    unittest.main()
