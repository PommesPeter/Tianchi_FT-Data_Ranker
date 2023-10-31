from lm_eval.base import Task, rf
from lm_eval.metrics import rouge


class QMSummarization(Task):
    VERSION = 1

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        for doc in self.dataset["train"]:
            yield from self._process_doc(doc)

    def test_docs(self):
        for doc in self.dataset["test"]:
            yield from self._process_doc(doc)

    def _process_doc(self, doc):
        input = doc["input"]
        split = input.find("\n\n")
        out_doc = {
            "id": doc["id"],
            "pid": doc["pid"],
            "input": input,
            "question": input[0:split],
            "text": input[split + 2:]
        }
        if "outputs" in doc:
            out_doc["outputs"] = doc["outputs"]
        return [out_doc]

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nQuestion: {doc['question']}\nAnswer:"

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
