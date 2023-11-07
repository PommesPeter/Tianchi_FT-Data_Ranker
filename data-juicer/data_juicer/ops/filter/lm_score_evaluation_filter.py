from jsonargparse.typing import ClosedUnitInterval
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import prepare_model, get_model

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('lm_score_evaluation_filter')
class LanguageModelEvaluationFilter(Filter):
    """Filter to keep samples in a specific language with confidence score
    larger than a specific min value."""

    def __init__(
        self,
        lang: str = '',
        min_score: ClosedUnitInterval = 0.8,
        dimension: str = 'helpfulness',
        model_name_or_path="gpt2",
        *args,
        **kwargs
    ):
        """
        Initialization method.

        :param lang: Samples in which language to keep.
        :param min_score: The min language identification confidence
            scores of samples to keep.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.dimension = dimension
        self.prompt = {
            "en": "We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.\nInstruction: [{}]\nInput: [{}]\nResponse: [{}]\nPlease rate according to the accuracy of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the helpfulness. Please first output a single line containing the value indicating the scores.",
            "zh": "请检查模型是否加载成功。如果没有，请稍后重试。",
        }

        from transformers import AutoModelForCausalLM

        logger.info("Loading language model from HuggingFace...")
        self.lm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def compute_stats(self, sample):
        # check if it's computed already
        if StatsKeys.language_model_score in sample[Fields.stats]:
            return sample

        text = sample[self.text_key].lower().replace('\n', ' ')
        # ft_model = get_model(self.model_key, lang=self.lang, model_type='fasttext')
        # TODO: using prompt and giving to the language model to predict text quality
        if self.lm_model is None:
            err_msg = 'Model not loaded. Please retry later.'
            logger.error(err_msg)
            raise ValueError(err_msg)
        # pred = ft_model.predict(text)
        # lang_id = pred[0][0].replace('__label__', '')
        # lang_score = pred[1][0]
        language_model_score = 0

        sample[Fields.stats][StatsKeys.language_model_score] = language_model_score

        return sample

    def process(self, sample):
        if sample[Fields.stats][StatsKeys.language_model_score] >= self.min_score:
            return True
        return False
