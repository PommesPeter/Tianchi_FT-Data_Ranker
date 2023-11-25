from ..base_op import OPERATORS, Selector


@OPERATORS.register_module('word_num_balanced_selector')
class WordNumBalancedSelector(Selector):
    """Selector to select samples based on the sorted frequency of specified
    field."""

    def __init__(self,
                 field_keys: str = '__dj__stats__.word_num',
                 min_range: int = 5,
                 max_range: int = 300,
                 interval: int = 20,
                 num_sample: int = 80000,
                 seed: int = 123,
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
        if self.text_key == 'instruction':
            self.field_keys = '__dj__stats__.' + self.text_key + '_num_words'
        self.min_range = min_range
        self.max_range = max_range
        self.interval = interval
        self.num_sample = num_sample
        self.seed = seed

        self.num_buckets = (self.max_range -
                            self.min_range) // self.interval + 1
        self.bucket_nums = [
            num_sample // self.num_buckets for _ in range(self.num_buckets)
        ]

    def process(self, dataset):
        if len(dataset) <= 1 or not self.field_keys:
            return dataset

        field_keys = self.field_keys.split('.')
        assert field_keys[0] in dataset.features.keys(
        ), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        buckets = [[] for _ in range(self.num_buckets)]
        for i, item in enumerate(dataset[field_keys[0]]):
            field_value = item
            for key in field_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(
                    key, field_value.keys())
                field_value = field_value[key]
            bucket_id = (field_value - self.min_range) // self.interval
            if bucket_id >= self.num_buckets:
                bucket_id = -1
            if len(buckets[bucket_id]) < self.bucket_nums[bucket_id]:
                buckets[bucket_id].append(i)

        select_index = []
        for id_list in buckets:
            print(len(id_list))
            select_index.extend(id_list)

        return dataset.select(select_index).shuffle(seed=self.seed)


if __name__ == '__main__':
    from datasets import load_dataset

    seletor = WordNumBalancedSelector(field_keys='__dj__stats__.num_words',
                                      text_key='instruction')
    data_path = '/home/xiejunlin/workspace/Tianchi_FT-Data_Ranker/checkpoints/ \
                 run/run_all_3sigma_v11_from_v4_en_2023-11-21-18-55-00/data/en\
                 /datasets_en.jsonl'

    ds_en = load_dataset('json', data_files=data_path, split='train')
    sample_ds = seletor.process(ds_en)
    print(sample_ds)
    sample_ds.to_json('word_num_sample.jsonl', force_ascii=False)
