from lm_eval.base import MultipleChoiceTask


class MultipleChoiceMassive(MultipleChoiceTask):
    VERSION = 1

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = "The following are multiple choice questions (with answers)."
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_example(doc, keys):
            question = doc["question"].strip()
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt = f"{question}\n{choices}Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        out_doc = {
            "query": format_example(doc, keys),
            "choices": keys,
        }
        if "answer" in doc:
            out_doc["gold"] = doc["answer"]
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

