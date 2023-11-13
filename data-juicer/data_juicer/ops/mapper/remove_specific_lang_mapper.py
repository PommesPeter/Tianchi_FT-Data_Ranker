from typing import List, Union

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_specific_languages_mapper')
class RemoveSpecificLanguagesMapper(Mapper):
    """Mapper to clean specific chars in text samples."""

    def __init__(self,
                 langs_to_remove: Union[str, List[str]] = [
                     'Japanese', 'Korean', 'Chinese'
                 ],
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param chars_to_remove: a list or a string including all
            characters that need to be removed from text.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        if langs_to_remove:
            unicode_to_remove = []
            for lang in langs_to_remove:
                if lang == 'Chinese':
                    unicode_to_remove.append('\u4e00-\u9fa5')
                elif lang == 'Korean':
                    unicode_to_remove.append('\uac00-\ud7ff')
                elif lang == 'Japanese':
                    unicode_to_remove.append('\u30a0-\u30ff\u3040-\u309f')

            self.pattern = u'[' + '|'.join(unicode_to_remove) + ']+'
        else:
            self.pattern = None

    def process(self, sample):

        if self.pattern is None:
            return sample

        sub_texts = re.findall(self.pattern, sample[self.text_key])
        for text in sub_texts:
            sample[self.text_key] = sample[self.text_key].replace(text, '')
        return sample
