import sys

from jsonargparse.typing import PositiveInt

from data_juicer.utils.constant import Fields

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_WORDS


@OPERATORS.register_module('specified_field_words_num_filter')
@INTER_WORDS.register_module('specified_field_words_num_filter')
class SpecifiedFieldWordNumFilter(Filter):
    """Filter to keep samples with total words number within a specific
    range."""

    def __init__(self,
                 min_num: PositiveInt = 10,
                 max_num: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: sample in which language.
        :param tokenization: whether to use model to tokenize documents
        :param min_num: The min filter word number in this op, samples
            will be filtered if their word number is below this
            parameter.
        :param max_num: The max filter word number in this op, samples
            will be filtered if their word number exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_num = min_num
        self.max_num = max_num

    def compute_stats(self, sample):
        # check if it's computed already
        if self.text_key + '_num_words' in sample[Fields.stats]:
            return sample
        words = sample[self.text_key].split(' ')
        sample[Fields.stats][self.text_key + '_num_words'] = len(words)
        return sample

    def process(self, sample):
        if self.min_num <= sample[Fields.stats][self.text_key +
                                                '_num_words'] <= self.max_num:
            return True
        else:
            return False
