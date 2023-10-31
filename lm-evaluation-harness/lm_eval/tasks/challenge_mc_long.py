import re

import numpy as np
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


def _normalize_answer(text):
    return " ".join(text.split()).strip()


class MultipleChoiceLong(Task):
    VERSION = 1

    _multiple_choice_pattern = re.compile(r" *\([A-D]\) *")

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

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nQuestion: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + ", ".join(doc["outputs"])
    
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

        split = out_doc["text"].find("\n\n", out_doc["text"].find("(D)"))
        choices_text = out_doc["text"][:split]
        out_doc["text"] = out_doc["text"][split:].strip()
        out_doc["choices"] = [_normalize_answer(choice) for choice in re.split(
            self._multiple_choice_pattern, choices_text)[1:]]
        
        if "outputs" in out_doc:
            out_doc["gold"] = out_doc["choices"].index(_normalize_answer(out_doc["outputs"][0]))
        
        return [out_doc]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {"acc": acc, "acc_norm": acc_norm}

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]
        return lls

    def aggregation(self):
        return {"acc": mean, "acc_norm": mean}

    def higher_is_better(self):
        return {"acc": True, "acc_norm": True}

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["input"]
