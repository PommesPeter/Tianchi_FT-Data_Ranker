from typing import List, Tuple, Union

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('keyword_mapper')
class KeywordMapper(Mapper):
    """Mapper to fix unicode errors in text samples."""

    def __init__(self,
                 keywords: Union[str, List[str], Tuple[str]] = ['지금 번역하기'],
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.keywords = keywords

    def process(self, sample):
        for keyword in self.keywords:
            sample[self.text_key] = sample[self.text_key].replace(keyword, '')
        return sample
