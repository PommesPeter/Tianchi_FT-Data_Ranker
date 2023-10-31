import transformers
from typing import Optional, Dict
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_path: Optional[str] = field(default="")


def convert_to_hf():

    parser = transformers.HfArgumentParser(
            (ModelArguments, TrainingArguments)
        )
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.lora_path,
        trust_remote_code=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, training_args.lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    convert_to_hf()