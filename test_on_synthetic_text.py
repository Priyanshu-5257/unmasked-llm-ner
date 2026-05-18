import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification
from transformers.models.lfm2.modeling_lfm2 import create_causal_mask, apply_mask_to_padding_states, DynamicCache
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
import types
import numpy as np

model_name = "LiquidAI/LFM2.5-350M"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
num_labels = len(label_list)

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

def patched_conv_forward(self, hidden_states, past_key_values=None, attention_mask=None):
    seqlen = hidden_states.shape[1]
    x = apply_mask_to_padding_states(hidden_states, attention_mask)
    BCx = self.in_proj(x).transpose(-1, -2)
    B, C, x = BCx.chunk(3, dim=-2)
    Bx = B * x
    conv_out_full = self.conv(Bx)
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
        self.base_model.forward = types.MethodType(patched_lfm_model_forward, self.base_model)

        num_layers = self.base_model.config.num_hidden_layers
        for i in range(num_layers - 8, num_layers):
            layer = self.base_model.layers[i]
            if hasattr(layer, 'conv'):
                layer.conv.forward = types.MethodType(patched_conv_forward, layer.conv)

        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

        hidden_size = self.base_model.config.hidden_size
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        ).to(self.base_model.dtype)

        for param in self.classifier.parameters():
            param.requires_grad = True

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
        return {"logits": logits}

model = LFMForTokenClassification(model_name, num_labels)
model.load_state_dict(load_file("./results_lfm_unmasked_lora_8layers/checkpoint-2634/model.safetensors"))
model.eval()
model.cuda()

with open("synthetic_test_text.txt", "r") as f:
    text = f.read()

print(f"Processing text: {len(text)} characters, {len(text.split())} words")

sentences = [s.strip() for s in text.replace('\n', ' ').split('. ') if s.strip()]
if not sentences[-1].endswith('.'):
    sentences[-1] += '.'
sentences = [s + '.' if not s.endswith('.') else s for s in sentences]

print(f"Split into {len(sentences)} sentences")

all_entities = []
for sent in sentences:
    tokenized = tokenizer(sent, truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokenized["input_ids"].cuda()
    attention_mask = tokenized["attention_mask"].cuda()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_ids = tokenized.word_ids(batch_index=0)
    
    # Build word-level labels by taking the first subword's prediction for each word
    word_labels = {}
    word_tokens = {}
    for i, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
        if word_idx is None:
            continue
        if word_idx not in word_labels:
            word_labels[word_idx] = pred
            word_tokens[word_idx] = []
        word_tokens[word_idx].append(tokens[i])
    
    # Merge subword tokens into full words
    words = []
    for word_idx in sorted(word_tokens.keys()):
        toks = word_tokens[word_idx]
        word = ""
        for t in toks:
            if t.startswith('Ġ'):
                word += t[1:]
            else:
                word += t
        words.append((word, word_labels[word_idx]))
    
    # Extract entities from word-level labels
    entities = []
    current_entity = []
    current_type = None
    
    for word_text, pred in words:
        label = label_list[pred]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_type, ' '.join(current_entity)))
            current_entity = [word_text]
            current_type = label[2:]
        elif label.startswith('I-') and current_type == label[2:]:
            current_entity.append(word_text)
        else:
            if current_entity:
                entities.append((current_type, ' '.join(current_entity)))
            current_entity = []
            current_type = None
    
    if current_entity:
        entities.append((current_type, ' '.join(current_entity)))
    
    if entities:
        all_entities.append((sent[:80] + "...", entities))

print("\n" + "=" * 80)
print("EXTRACTED ENTITIES FROM SYNTHETIC TEXT")
print("=" * 80)

entity_counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
for sent_preview, entities in all_entities:
    print(f"\nSentence: {sent_preview}")
    for ent_type, ent_text in entities:
        print(f"  [{ent_type}] {ent_text}")
        entity_counts[ent_type] = entity_counts.get(ent_type, 0) + 1

print("\n" + "=" * 80)
print("ENTITY COUNTS")
print("=" * 80)
for ent_type, count in entity_counts.items():
    print(f"  {ent_type}: {count}")
print(f"  Total: {sum(entity_counts.values())}")
