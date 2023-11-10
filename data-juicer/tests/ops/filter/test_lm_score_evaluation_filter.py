import unittest

from datasets import Dataset

from data_juicer.ops.filter.lm_score_evaluation_filter import \
    LanguageModelEvaluationFilter
from data_juicer.utils.constant import Fields


class MaximumLineLengthFilterTest(unittest.TestCase):

    def _run_lm_score_evaluation_filter(self, dataset: Dataset, target_list,
                                        op):
        if Fields.stats not in dataset.features:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{"instruction":"where did all the money that was lost in the sub prime mortgage crisis and subsequent global recession actually go ? Money does n't simply disappear so if people lost money then it must have gone somewhere . Where ? Explain like I'm five.","input":"","output":"The subprime mortgage crisis was a financial crisis that occurred in the late 2000s as a result of the collapse of the housing market. Many people had taken out mortgages to buy homes, but when the housing market crashed, the value of their homes decreased and they were no longer able to make their mortgage payments. This led to a wave of foreclosures, which is when the bank takes back a home because the owner can't make the payments. As a result, many banks and financial institutions lost a lot of money because they had invested in these mortgages and were now unable to recoup their investments. This caused a financial crisis, as the banks and financial institutions that had lost money were no longer able to lend money to other people or businesses. The money that was lost during the financial crisis did not simply disappear. Instead, it was lost by the banks and financial institutions that had invested in the subprime mortgages and were unable to recoup their investments when the housing market crashed. This loss of money had a ripple effect, as the banks and financial institutions were then unable to lend money to other people or businesses, which slowed down the economy and led to a global recession.", "text": "where did all the money that was lost in the sub prime mortgage crisis and subsequent global recession actually go ? Money does n't simply disappear so if people lost money then it must have gone somewhere . Where ? Explain like I'm five. The subprime mortgage crisis was a financial crisis that occurred in the late 2000s as a result of the collapse of the housing market. Many people had taken out mortgages to buy homes, but when the housing market crashed, the value of their homes decreased and they were no longer able to make their mortgage payments. This led to a wave of foreclosures, which is when the bank takes back a home because the owner can't make the payments. As a result, many banks and financial institutions lost a lot of money because they had invested in these mortgages and were now unable to recoup their investments. This caused a financial crisis, as the banks and financial institutions that had lost money were no longer able to lend money to other people or businesses. The money that was lost during the financial crisis did not simply disappear. Instead, it was lost by the banks and financial institutions that had invested in the subprime mortgages and were unable to recoup their investments when the housing market crashed. This loss of money had a ripple effect, as the banks and financial institutions were then unable to lend money to other people or businesses, which slowed down the economy and led to a global recession."}]
        tgt_list = [{"text": "where did all the money that was lost in the sub prime mortgage crisis and subsequent global recession actually go ? Money does n't simply disappear so if people lost money then it must have gone somewhere . Where ? Explain like I'm five. The subprime mortgage crisis was a financial crisis that occurred in the late 2000s as a result of the collapse of the housing market. Many people had taken out mortgages to buy homes, but when the housing market crashed, the value of their homes decreased and they were no longer able to make their mortgage payments. This led to a wave of foreclosures, which is when the bank takes back a home because the owner can't make the payments. As a result, many banks and financial institutions lost a lot of money because they had invested in these mortgages and were now unable to recoup their investments. This caused a financial crisis, as the banks and financial institutions that had lost money were no longer able to lend money to other people or businesses. The money that was lost during the financial crisis did not simply disappear. Instead, it was lost by the banks and financial institutions that had invested in the subprime mortgages and were unable to recoup their investments when the housing market crashed. This loss of money had a ripple effect, as the banks and financial institutions were then unable to lend money to other people or businesses, which slowed down the economy and led to a global recession."}]
        dataset = Dataset.from_list(ds_list)
        op = LanguageModelEvaluationFilter(lang='en', min_score=4, cuda_device="cuda:1")
        self._run_lm_score_evaluation_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
