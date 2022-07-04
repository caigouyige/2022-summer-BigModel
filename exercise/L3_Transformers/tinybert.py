# Load the dataset

from datasets import load_dataset, load_metric
dataset = load_dataset("glue", 'qnli')
metric = load_metric('glue', 'qnli')

# Tokenization

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=50, add_special_tokens = True)
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tune

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

from transformers import TrainingArguments

args = TrainingArguments(
    "tiny-bert-finetuned-qnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()