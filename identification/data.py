from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

# Configuration - 12 Low-Resource Languages
languages = [
    "mt",  # Maltese
    "af",  # Afrikaans  
    "is",  # Icelandic
    "cy",  # Welsh
    "mk",  # Macedonian
    "lv",  # Latvian
    "lt",  # Lithuanian
    "sl",  # Slovenian
    "sk",  # Slovak
    "et",  # Estonian
    "ka",  # Georgian
    "ne"   # Nepali
]

target_size_mb = 100  # Total size in MB for train + valid
val_ratio = 0.02
save_dir = "data_nemo" 

# FLORES-200 language code mappings for low-resource languages
flores_lang_mapping = {
    'mt': 'mlt_Latn',  # Maltese
    'af': 'afr_Latn',  # Afrikaans
    'is': 'isl_Latn',  # Icelandic
    'cy': 'cym_Latn',  # Welsh
    'mk': 'mkd_Cyrl',  # Macedonian
    'lv': 'lvs_Latn',  # Latvian
    'lt': 'lit_Latn',  # Lithuanian
    'sl': 'slv_Latn',  # Slovenian
    'sk': 'slk_Latn',  # Slovak
    'et': 'ekk_Latn',  # Estonian
    'ka': 'kat_Geor',  # Georgian
    'ne': 'npi_Deva',  # Nepali
}

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Base-2407", use_fast=True)

# Convert MB to bytes
target_total_bytes = target_size_mb * 1024 * 1024

for lang in languages:
    print(f"Processing language: {lang}")
    language_code = flores_lang_mapping.get(lang, lang)
    language_code = language_code.replace('_', '-')
    
    try:
        ds_streamed = load_dataset("cis-lmu/GlotCC-V1", language_code, split="train", streaming=True, use_auth_token=True)
    except Exception as e:
        print(f"Error loading {lang} ({language_code}): {e}")
        continue

    train_ids, val_ids = [], []
    total_bytes = 0
    split_point = int((1 - val_ratio) * target_total_bytes)

    for item in ds_streamed:
        text = item["content"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        byte_tensor = torch.LongTensor(ids).numpy().tobytes()
        size_bytes = len(byte_tensor)

        if total_bytes + size_bytes > target_total_bytes:
            break

        if total_bytes < split_point:
            train_ids.extend(ids)
        else:
            val_ids.extend(ids)

        total_bytes += size_bytes

    def save_tensor(tensor_ids, split_name):
        tensor = torch.LongTensor(tensor_ids)
        save_path = os.path.join(save_dir, f"id.{lang}.{split_name}.nemo")
        torch.save(tensor, save_path)
        print(f"Saved {split_name} ({len(tensor)} tokens, ~{tensor.numpy().nbytes / 1024 / 1024:.2f} MB) to {save_path}")

    save_tensor(train_ids, "train")
    save_tensor(val_ids, "valid")
    print(f"Completed processing for {lang}")
    print("-" * 50)