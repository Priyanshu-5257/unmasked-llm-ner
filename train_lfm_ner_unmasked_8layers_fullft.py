import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorForTokenClassification
from transformers.models.lfm2.modeling_lfm2 import create_causal_mask, apply_mask_to_padding_states, DynamicCache
from datasets import load_dataset
import numpy as np
import evaluate
import json
import os
import types

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

# Define patched forward for Lfm2Model to disable causal mask on last 2 layers
def patched_lfm_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    linear_attention = attention_mask if inputs_embeds.shape[1] != 1 else None

    # Create expanded non-causal mask for full_attention layers
    if attention_mask is not None:
        expanded_non_causal_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
        expanded_non_causal_mask = (1.0 - expanded_non_causal_mask) * torch.finfo(inputs_embeds.dtype).min
        expanded_non_causal_mask = expanded_non_causal_mask.expand(-1, -1, inputs_embeds.shape[1], -1)
    else:
        expanded_non_causal_mask = None

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    num_layers = self.config.num_hidden_layers
    for i, decoder_layer in enumerate(self.layers[:num_layers]):
        # UNMASK LAST 8 LAYERS
        is_attention = self.config.layer_types[i] == "full_attention"
        if i >= num_layers - 8:
            layer_mask = expanded_non_causal_mask if is_attention else linear_attention
        else:
            layer_mask = causal_mask if is_attention else linear_attention

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = self.embedding_norm(hidden_states)
    return {"last_hidden_state": hidden_states, "past_key_values": past_key_values}

# Define patched forward for Lfm2ShortConv to disable causality
def patched_conv_forward(self, hidden_states, past_key_values=None, attention_mask=None):
    seqlen = hidden_states.shape[1]
    x = apply_mask_to_padding_states(hidden_states, attention_mask)
    BCx = self.in_proj(x).transpose(-1, -2)
    B, C, x = BCx.chunk(3, dim=-2)
    Bx = B * x

    conv_out_full = self.conv(Bx)
    
    # non-causal (centered) slice
    left_pad = self.L_cache // 2
    conv_out = conv_out_full[..., left_pad : left_pad + seqlen]

    y = C * conv_out
    y = y.transpose(-1, -2).contiguous()
    y = self.out_proj(y)
    return y


class LFMForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Patch the model's forward
        self.base_model.forward = types.MethodType(patched_lfm_model_forward, self.base_model)
        
        # Patch the conv layers in the last 8 layers if they exist
        num_layers = self.base_model.config.num_hidden_layers
        for i in range(num_layers - 8, num_layers):
            layer = self.base_model.layers[i]
            if hasattr(layer, 'conv'):
                layer.conv.forward = types.MethodType(patched_conv_forward, layer.conv)

        # Freeze base model, except last 8 layers
        for name, param in self.base_model.named_parameters():
            # Check if name contains any layer index from num_layers-8 to num_layers-1
            unfrozen = any(f"layers.{i}" in name for i in range(num_layers - 8, num_layers))
            if unfrozen:
                param.requires_grad = True
            else:
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
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if isinstance(outputs, dict) and 'last_hidden_state' in outputs:
            sequence_output = outputs['last_hidden_state']
        elif hasattr(outputs, 'last_hidden_state'):
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
    output_dir="./results_lfm_unmasked_8layers_fullft",
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

print("Starting training for Custom LFM NER (Unmasked Last 8 Layers Full FT)...")
trainer.train()

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)

with open("lfm_unmasked_8layers_fullft_test_results.json", "w") as f:
    json.dump(test_results, f)
