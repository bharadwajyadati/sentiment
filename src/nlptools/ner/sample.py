from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-large-finetuned-conll03-english")
print(model)

model.predict(" UAE Is working on new project")
