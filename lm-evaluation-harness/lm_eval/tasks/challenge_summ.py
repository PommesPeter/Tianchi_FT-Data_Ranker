from lm_eval.base import Task, rf
from lm_eval.metrics import rouge


class Summarization(Task):
    VERSION = 1

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"{doc['input']}\n\nQuestion: What is a summary of the preceding text?\nAnswer:"
   
    def doc_to_target(self, doc):
        return " " + ", ".join(doc["outputs"])

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["outputs"]),
            "rouge2": (results[0], doc["outputs"]),
            "rougeL": (results[0], doc["outputs"]),
        }

    def construct_requests(self, doc, ctx):
        return [rf.greedy_until(ctx, {'until': ["\n"]})]

    def _make_compute_metrics(self, value):
        def compute_metrics(samples):
            predictions, references = zip(*samples)
            computed = rouge(predictions=predictions, references=references)
            return computed[value]
        return compute_metrics

    def aggregation(self):
        return {
            "rouge1": self._make_compute_metrics("rouge1"),
            "rouge2": self._make_compute_metrics("rouge2"),
            "rougeL": self._make_compute_metrics("rougeL"),
        }

    def higher_is_better(self):
        return {"rouge1": True, "rouge2": True, "rougeL": True}

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["input"]
