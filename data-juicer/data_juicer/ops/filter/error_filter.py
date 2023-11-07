from typing import List, Tuple, Union

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('error_filter')
class ErrorFilter(Filter):
    """Filter to keep samples with specified suffix."""

    def __init__(self,
                 errors: Union[str, List[str], Tuple[str]] = [],
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param suffixes: the suffix of text that will be keep.
            For example: '.txt', 'txt' or ['txt', '.pdf', 'docx']
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if errors is None:
            self.errors = []
        elif isinstance(errors, str):
            self.errors = [errors]
        else:
            self.errors = errors

    def compute_stats(self, sample):
        return sample

    def process(self, sample):
        if self.errors:
            for error in self.errors:
                if error in sample[self.text_key]:
                    return False
            return True
        else:
            return True
