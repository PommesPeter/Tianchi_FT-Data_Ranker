from lm_eval.base import MultipleChoiceTask


class MultipleChoiceZH(MultipleChoiceTask):
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
        description= "以下是单项选择题，请直接给出正确答案的选项。"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_example(doc, keys):
            question = doc["Question"].strip()
            choices = "".join(
                [f'{key}. {doc[key]}\n' for key in keys]
            )
            prompt = f"{question}\n{choices}答案："
            return prompt
        keys = ["A", "B", "C", "D"]
        out_doc = {
            "query": format_example(doc, keys),
            "choices": keys,
        }
        if "Answer" in doc:
            out_doc["gold"] = ord(doc["Answer"])-ord("A")
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
