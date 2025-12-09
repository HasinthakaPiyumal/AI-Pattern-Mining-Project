import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler
import random
import collections
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
MODEL_NAME = "facebook/bart-base"
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS_PRETRAIN = 1
NUM_WARMUP_STEPS = 0
ACCUMULATION_STEPS = 2

# --- Denoising Functions ---
def mask_tokens(inputs, tokenizer, mask_ratio=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mask_ratio)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The remaining 10% of the time, keep the original word
    return inputs, labels

def delete_tokens(inputs, tokenizer, delete_ratio=0.1):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, delete_ratio)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    delete_indices = torch.bernoulli(probability_matrix).bool()

    new_inputs = []
    new_labels = []
    for i in range(inputs.shape[0]):
        seq_inputs = inputs[i][~delete_indices[i]].tolist()
        seq_labels = labels[i].tolist()
        # Simple handling: pad to original length for consistent batching
        padded_inputs = seq_inputs + [tokenizer.pad_token_id] * (inputs.shape[1] - len(seq_inputs))
        new_inputs.append(padded_inputs)
        new_labels.append(seq_labels)
    
    return torch.tensor(new_inputs), torch.tensor(new_labels)

def infill_spans(inputs, tokenizer, span_mask_ratio=0.15, min_span_length=2, max_span_length=10):
    labels = inputs.clone()
    seq_len = inputs.shape[1]
    
    input_ids = inputs.tolist()
    original_labels = labels.tolist()

    corrupted_inputs_batch = []
    target_labels_batch = []

    for batch_idx in range(inputs.shape[0]):
        current_input_ids = list(input_ids[batch_idx])
        current_labels_ids = list(original_labels[batch_idx])
        
        # Find non-special token indices
        non_special_indices = [i for i, token_id in enumerate(current_input_ids) 
                               if token_id not in tokenizer.all_special_ids and token_id != -100]
        
        if not non_special_indices: # Handle empty or all-special sequences
            corrupted_inputs_batch.append(current_input_ids)
            target_labels_batch.append(current_labels_ids)
            continue

        num_to_mask = int(len(non_special_indices) * span_mask_ratio)
        if num_to_mask == 0: # Ensure at least one token is masked if possible
            if len(non_special_indices) > 0: num_to_mask = 1
            else: 
                corrupted_inputs_batch.append(current_input_ids)
                target_labels_batch.append(current_labels_ids)
                continue

        masked_indices = set()
        while len(masked_indices) < num_to_mask:
            start_idx = random.choice(non_special_indices)
            span_length = random.randint(min_span_length, max_span_length)
            
            current_span = []
            for i in range(start_idx, min(start_idx + span_length, seq_len)):
                if i in non_special_indices and i not in masked_indices:
                    current_span.append(i)
            
            if current_span:
                for idx in current_span:
                    if len(masked_indices) < num_to_mask:
                        masked_indices.add(idx)
                    else:
                        break
        
        corrupted_input_tokens = []
        target_label_tokens = []
        i = 0
        while i < seq_len:
            if i in masked_indices:
                corrupted_input_tokens.append(tokenizer.mask_token_id)
                # Collect all original tokens from the masked span for the target label
                j = i
                while j < seq_len and j in masked_indices:
                    target_label_tokens.append(current_labels_ids[j])
                    j += 1
                i = j
            else:
                corrupted_input_tokens.append(current_input_ids[i])
                target_label_tokens.append(-100) # Only predict masked tokens
                i += 1
        
        # Pad the target labels to a fixed length or handle dynamically later
        # For simplicity, let's just make sure input and target have a corresponding structure.
        # In BART, the decoder receives the full original sequence for the target, 
        # but loss is only computed for masked parts. Here we're simplifying this.
        # A more faithful BART infilling would rebuild the target from original tokens + sentinel IDs.
        # For this example, we'll keep it simple: input has [MASK], target has original token for masked span, -100 elsewhere
        # This simplified `target_label_tokens` will effectively serve as the decoder_output_ids
        # when the model is trained with a denoising objective where it reconstructs the original.
        # The actual BART pretraining uses sentinel tokens and reconstructs them.
        # Given the 