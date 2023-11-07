import numbers
import random

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module('specified_source_selector')
class SpecifiedSourceFieldSelector(Selector):
    """Selector to select samples based on the sorted frequency of specified
    field."""

    def __init__(self,
                 field_keys: str = '',
                 num_sample: int = 10000,
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param field_keys: Selector based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_keys = field_keys
        self.num_sample = num_sample

    def process(self, dataset):
        if len(dataset) <= 1 or not self.field_keys:
            return dataset

        field_keys = self.field_keys.split('.')
        assert field_keys[0] in dataset.features.keys(
        ), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        field_value_dict = {}
        for i, item in enumerate(dataset[field_keys[0]]):
            field_value = item
            for key in field_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(
                    key, field_value.keys())
                field_value = field_value[key]
            assert field_value is None or isinstance(
                field_value, str) or isinstance(
                    field_value, numbers.Number
                ), 'The {} item is not String, Numbers or NoneType'.format(i)
            if field_value not in field_value_dict.keys():
                field_value_dict[field_value] = [i]
            else:
                field_value_dict[field_value].append(i)

        select_index = []
        random.seed(123)
        for key, field_value in field_value_dict.items():
            if len(field_value) > self.num_sample:
                sample_index = random.sample(field_value, self.num_sample)
                select_index.extend(sample_index)
            else:
                select_index.extend(field_value)

        return dataset.select(select_index).shuffle(seed=123)


if __name__ == "__main__":
    from datasets import load_dataset
    seletor = SpecifiedSourceFieldSelector(field_keys='meta.Dataset')
    data_path = "/home/xiejunlin/workspace/Tianchi_FT-Data_Ranker/checkpoints/run/run_keep_long_filter_token_perplexity_en05zh05_2023-10-30-21-47-33/data/en/datasets_en.jsonl"
    ds_en = load_dataset('json', data_files=data_path, split='train')
    sample_ds = seletor.process(ds_en)
    print(sample_ds[0])
