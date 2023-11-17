import argostranslate.translate
import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_specific_languages_mapper')
class RemoveSpecificLanguagesMapper(Mapper):
    """Mapper to clean specific chars in text samples."""

    def __init__(self, langs_to_remove: str = 'Japanese', *args, **kwargs):
        """
        Initialization method.

        :param chars_to_remove: a list or a string including all
            characters that need to be removed from text.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        if langs_to_remove:
            if langs_to_remove == 'Chinese':
                self.from_code = 'zh'
                self.to_code = 'en'
                unicode_to_remove = '\u4e00-\u9fa5'
            elif langs_to_remove == 'Korean':
                self.from_code = 'ko'
                self.to_code = 'en'
                unicode_to_remove = '\uac00-\ud7ff'
            elif langs_to_remove == 'Japanese':
                self.from_code = 'ja'
                self.to_code = 'en'
                unicode_to_remove = '\u30a0-\u30ff\u3040-\u309f'

            self.pattern = u'[' + unicode_to_remove + ']+'
        else:
            self.pattern = None

    def process(self, sample):

        if self.pattern is None:
            return sample

        matches = re.finditer(self.pattern, sample[self.text_key])
        idxs = []
        for match in matches:
            start = match.start()
            end = match.end()
            idxs.append((start, end))
        if len(idxs):
            sub_text = sample[self.text_key][idxs[0][0]:idxs[-1][-1]]
            translate_text = argostranslate.translate.translate(
                sub_text, self.from_code, self.to_code)
            sample[self.text_key] = sample[self.text_key].replace(
                sub_text, ' ' + translate_text)

        return sample
