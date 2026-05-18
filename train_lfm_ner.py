import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np
import evaluate
import json
import os

os.environ["WANDB_DISABLED"] = "true"

model_name = "LiquidAI/LFM2.5-350M"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load CoNLL-2003 dataset
dataset = load_dataset("conll2003", trust_remote_code=True)
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

class LFMForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        hidden_size = self.base_model.config.hidden_size
        self.num_labels = num_labels
        
        # Custom head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        ).to(self.base_model.dtype)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # The base LFM might not need attention_mask if it's an RNN/SSM, but we pass it if we have it
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # outputs usually has last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs
            
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

model = LFMForTokenClassification(model_name, num_labels)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="./results_lfm",
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training for Custom LFM NER...")
trainer.train()

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)

with open("lfm_test_results.json", "w") as f:
    json.dump(test_results, f)
