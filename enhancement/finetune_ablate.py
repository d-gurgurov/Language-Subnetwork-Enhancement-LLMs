import argparse
import json
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt

class LanguageSpecificDataset(Dataset):
    """Dataset for fine-tuning on target language text chunks"""
    
    def __init__(self, text_chunks, tokenizer, max_length=512):
        self.text_chunks = text_chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        text = self.text_chunks[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze() 
        }

class RandomNeuronFineTuner:
    """Fine-tunes a random selection of neurons with the same count as target language neurons"""
    
    def __init__(self, model, tokenizer, activation_masks, target_lang, lang_to_idx, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_masks = activation_masks
        self.target_lang = target_lang
        self.lang_to_idx = lang_to_idx
        
        # Get target language neuron indices to match the count
        if target_lang in lang_to_idx:
            target_neuron_indices = activation_masks[lang_to_idx[target_lang]]

        # Generate random neuron indices with the same structure
        random.seed(seed)
        torch.manual_seed(seed)
        self.target_neuron_indices = self._generate_random_indices(target_neuron_indices)
        
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the random neurons
        self.unfreeze_target_neurons()
        
    def _generate_random_indices(self, target_indices):
        """Generate random neuron indices with the same overall count as target_indices, 
        but not preserving per-layer counts."""
        # Step 1: Figure out total number of neurons in each layer
        layer_neuron_spaces = []
        for layer_idx in range(len(target_indices)):
            mlp_layer = self.model.model.layers[layer_idx].mlp
            if hasattr(mlp_layer, 'gate_up_proj'):
                total_neurons = mlp_layer.gate_up_proj.weight.shape[0] // 2
            elif hasattr(mlp_layer, 'gate_proj'):
                total_neurons = mlp_layer.gate_proj.weight.shape[0]
            else:
                raise ValueError(f"Unknown MLP architecture at layer {layer_idx}")
            layer_neuron_spaces.append(total_neurons)
        
        # Step 2: Count total target neurons across all layers
        total_target_neurons = sum(len(neuron_indices) for neuron_indices in target_indices)
        
        # Step 3: Create a flat list of (layer_idx, neuron_id) for all possible neurons
        all_possible_neurons = [
            (layer_idx, neuron_id)
            for layer_idx, total_neurons in enumerate(layer_neuron_spaces)
            for neuron_id in range(total_neurons)
        ]
        
        # Step 4: Sample same total number of neurons randomly from the whole set
        sampled = random.sample(all_possible_neurons, min(total_target_neurons, len(all_possible_neurons)))
        
        # Step 5: Redistribute sampled neurons into layer-wise structure
        random_indices = [[] for _ in range(len(layer_neuron_spaces))]
        for layer_idx, neuron_id in sampled:
            random_indices[layer_idx].append(neuron_id)
        
        # Convert each to a tensor
        random_indices = [
            torch.tensor(indices, dtype=torch.long) if indices else []
            for indices in random_indices
        ]

        # Print the number of neurons chosen for each layer
        print(f"Neurons selected per layer:")
        for layer_idx, indices in enumerate(random_indices):
            num_neurons = len(indices) if isinstance(indices, list) else len(indices) if len(indices) > 0 else 0
            print(f"  Layer {layer_idx}: {num_neurons} neurons")
        
        return random_indices

    
    def unfreeze_target_neurons(self):
        """Unfreeze only the random neurons"""
        print(f"Unfreezing random neurons (matching count for {self.target_lang})")
        
        unfrozen_params = 0
        total_target_neurons = 0
        self.target_params = []
        
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        for layer_idx, neuron_indices in enumerate(self.target_neuron_indices):
            if len(neuron_indices) == 0:
                continue
                
            neuron_indices = neuron_indices.cpu()
            mlp_layer = self.model.model.layers[layer_idx].mlp
            
            # Same logic as LanguageNeuronFineTuner but with random indices
            if hasattr(mlp_layer, 'gate_up_proj'):
                gate_up_weight = mlp_layer.gate_up_proj.weight
                
                if hasattr(gate_up_weight, 'to_local'):
                    gate_up_local = gate_up_weight.to_local()
                    gate_target_data = gate_up_local[neuron_indices, :].clone()
                    hidden_size = gate_up_local.shape[0] // 2
                    up_indices = neuron_indices + hidden_size
                    up_target_data = gate_up_local[up_indices, :].clone()
                else:
                    gate_target_data = gate_up_weight[neuron_indices, :].clone()
                    hidden_size = gate_up_weight.shape[0] // 2
                    up_indices = neuron_indices + hidden_size
                    up_target_data = gate_up_weight[up_indices, :].clone()
                
                gate_target_weights = nn.Parameter(gate_target_data.to(model_device, dtype=model_dtype))
                up_target_weights = nn.Parameter(up_target_data.to(model_device, dtype=model_dtype))
                
                mlp_layer._target_gate_indices = neuron_indices
                mlp_layer._target_up_indices = up_indices
                mlp_layer._target_gate_weights = gate_target_weights
                mlp_layer._target_up_weights = up_target_weights
                mlp_layer._architecture_type = "combined"
                
                self.target_params.extend([gate_target_weights, up_target_weights])
                unfrozen_params += gate_target_weights.numel() + up_target_weights.numel()
                
            elif hasattr(mlp_layer, 'gate_proj') and hasattr(mlp_layer, 'up_proj'):
                gate_weight = mlp_layer.gate_proj.weight
                if hasattr(gate_weight, 'to_local'):
                    gate_target_data = gate_weight.to_local()[neuron_indices, :].clone()
                else:
                    gate_target_data = gate_weight[neuron_indices, :].clone()
                
                gate_target_weights = nn.Parameter(gate_target_data.to(model_device, dtype=model_dtype))
                mlp_layer._target_gate_indices = neuron_indices
                mlp_layer._target_gate_weights = gate_target_weights
                
                up_weight = mlp_layer.up_proj.weight
                if hasattr(up_weight, 'to_local'):
                    up_target_data = up_weight.to_local()[neuron_indices, :].clone()
                else:
                    up_target_data = up_weight[neuron_indices, :].clone()
                
                up_target_weights = nn.Parameter(up_target_data.to(model_device, dtype=model_dtype))
                mlp_layer._target_up_indices = neuron_indices
                mlp_layer._target_up_weights = up_target_weights
                mlp_layer._architecture_type = "separate"
                
                self.target_params.extend([gate_target_weights, up_target_weights])
                unfrozen_params += gate_target_weights.numel() + up_target_weights.numel()
                
            else:
                raise ValueError(f"Unknown MLP architecture at layer {layer_idx}")
            
            # Handle down_proj weights
            down_weight = mlp_layer.down_proj.weight
            if hasattr(down_weight, 'to_local'):
                down_target_data = down_weight.to_local()[:, neuron_indices].clone()
            else:
                down_target_data = down_weight[:, neuron_indices].clone()
                
            down_target_weights = nn.Parameter(down_target_data.to(model_device, dtype=model_dtype))
            mlp_layer._target_down_indices = neuron_indices
            mlp_layer._target_down_weights = down_target_weights
            
            self.target_params.append(down_target_weights)
            unfrozen_params += down_target_weights.numel()
            
            total_target_neurons += len(neuron_indices)
            
            # Use the same custom forward method as LanguageNeuronFineTuner
            mlp_layer._original_forward = mlp_layer.forward
            mlp_layer.forward = self.create_custom_mlp_forward(mlp_layer)
        
        print(f"Created {unfrozen_params:,} trainable parameters across {total_target_neurons} random neurons")
        return unfrozen_params
    
    def create_custom_mlp_forward(self, mlp_layer):
        """Create a custom forward function that uses our target parameters in the computation"""
        def custom_forward(hidden_states):
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            if mlp_layer._architecture_type == "combined":
                gate_up_weight = mlp_layer.gate_up_proj.weight
                
                if hasattr(gate_up_weight, 'to_local'):
                    gate_up_local = gate_up_weight.to_local().clone()
                else:
                    gate_up_local = gate_up_weight.clone()
                
                gate_up_local[mlp_layer._target_gate_indices, :] = mlp_layer._target_gate_weights.to(device, dtype=dtype)
                gate_up_local[mlp_layer._target_up_indices, :] = mlp_layer._target_up_weights.to(device, dtype=dtype)
                
                gate_up = F.linear(hidden_states, gate_up_local, 
                                 mlp_layer.gate_up_proj.bias.to_local() if hasattr(mlp_layer.gate_up_proj.bias, 'to_local') 
                                 else mlp_layer.gate_up_proj.bias)
                
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate_states = F.silu(gate) * up
                
            elif mlp_layer._architecture_type == "separate":
                gate_weight = mlp_layer.gate_proj.weight
                up_weight = mlp_layer.up_proj.weight
                
                if hasattr(gate_weight, 'to_local'):
                    gate_local = gate_weight.to_local().clone()
                else:
                    gate_local = gate_weight.clone()
                    
                if hasattr(up_weight, 'to_local'):
                    up_local = up_weight.to_local().clone()
                else:
                    up_local = up_weight.clone()
                
                gate_local[mlp_layer._target_gate_indices, :] = mlp_layer._target_gate_weights.to(device, dtype=dtype)
                up_local[mlp_layer._target_up_indices, :] = mlp_layer._target_up_weights.to(device, dtype=dtype)
                
                gate = F.linear(hidden_states, gate_local, 
                              mlp_layer.gate_proj.bias.to_local() if hasattr(mlp_layer.gate_proj.bias, 'to_local')
                              else mlp_layer.gate_proj.bias)
                up = F.linear(hidden_states, up_local,
                            mlp_layer.up_proj.bias.to_local() if hasattr(mlp_layer.up_proj.bias, 'to_local')
                            else mlp_layer.up_proj.bias)
                intermediate_states = F.silu(gate) * up
            
            down_weight = mlp_layer.down_proj.weight
            if hasattr(down_weight, 'to_local'):
                down_local = down_weight.to_local().clone()
            else:
                down_local = down_weight.clone()
                
            down_local[:, mlp_layer._target_down_indices] = mlp_layer._target_down_weights.to(device, dtype=dtype)
            
            output = F.linear(intermediate_states, down_local,
                            mlp_layer.down_proj.bias.to_local() if hasattr(mlp_layer.down_proj.bias, 'to_local')
                            else mlp_layer.down_proj.bias)
            
            return output
        
        return custom_forward
    
    def install_hooks(self):
        """Install custom forward methods for training"""
        for layer_idx, neuron_indices in enumerate(self.target_neuron_indices):
            if len(neuron_indices) == 0:
                continue
                
            mlp_layer = self.model.model.layers[layer_idx].mlp
            
            if not hasattr(mlp_layer, '_original_forward'):
                mlp_layer._original_forward = mlp_layer.forward
            
            mlp_layer.forward = self.create_custom_mlp_forward(mlp_layer)
    
    def remove_hooks(self):
        """Restore original forward methods"""
        for layer_idx in range(len(self.target_neuron_indices)):
            mlp_layer = self.model.model.layers[layer_idx].mlp
            if hasattr(mlp_layer, '_original_forward'):
                mlp_layer.forward = mlp_layer._original_forward

def integrate_trained_parameters(model, fine_tuner):
    """
    Integrate the trained target parameters back into the main model weights
    This is CRITICAL for save_pretrained to work correctly
    """
    
    print("Integrating trained parameters into model state...")
    
    with torch.no_grad():
        for layer_idx, neuron_indices in enumerate(fine_tuner.target_neuron_indices):
            if len(neuron_indices) == 0:
                continue
                
            mlp_layer = model.model.layers[layer_idx].mlp
            
            if not hasattr(mlp_layer, '_target_gate_weights'):
                continue
            
            if hasattr(mlp_layer, 'gate_up_proj'):
                model_device = mlp_layer.gate_up_proj.weight.device
                model_dtype = mlp_layer.gate_up_proj.weight.dtype
            else:
                model_device = mlp_layer.gate_proj.weight.device
                model_dtype = mlp_layer.gate_proj.weight.dtype
            
            if mlp_layer._architecture_type == "combined":
                gate_up_weight = mlp_layer.gate_up_proj.weight
                
                if hasattr(gate_up_weight, 'to_local'):
                    gate_up_local = gate_up_weight.to_local()
                    gate_up_local[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=gate_up_local.device, dtype=gate_up_local.dtype
                    )
                    gate_up_local[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=gate_up_local.device, dtype=gate_up_local.dtype
                    )
                else:
                    gate_up_weight.data[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=model_device, dtype=model_dtype
                    )
                    gate_up_weight.data[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=model_device, dtype=model_dtype
                    )
                    
            elif mlp_layer._architecture_type == "separate":
                gate_weight = mlp_layer.gate_proj.weight
                up_weight = mlp_layer.up_proj.weight
                
                if hasattr(gate_weight, 'to_local'):
                    gate_weight.to_local()[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=gate_weight.device, dtype=gate_weight.dtype
                    )
                    up_weight.to_local()[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=up_weight.device, dtype=up_weight.dtype
                    )
                else:
                    gate_weight.data[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=model_device, dtype=model_dtype
                    )
                    up_weight.data[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=model_device, dtype=model_dtype
                    )
            
            # Update down_proj weights
            down_weight = mlp_layer.down_proj.weight
            if hasattr(down_weight, 'to_local'):
                down_weight.to_local()[:, mlp_layer._target_down_indices] = mlp_layer._target_down_weights.to(
                    device=down_weight.device, dtype=down_weight.dtype
                )
            else:
                down_weight.data[:, mlp_layer._target_down_indices] = mlp_layer._target_down_weights.to(
                    device=model_device, dtype=model_dtype
                )
    
    print("Parameter integration complete!")

def clean_model_for_saving(model, fine_tuner):
    """
    Remove all custom attributes and parameters that shouldn't be saved
    """
    print("Cleaning model for saving...")
    
    for layer_idx, neuron_indices in enumerate(fine_tuner.target_neuron_indices):
        if len(neuron_indices) == 0:
            continue
            
        mlp_layer = model.model.layers[layer_idx].mlp
        
        # Remove custom attributes
        attrs_to_remove = [
            '_target_gate_indices', '_target_up_indices', '_target_down_indices',
            '_target_gate_weights', '_target_up_weights', '_target_down_weights',
            '_architecture_type', '_original_forward'
        ]
        
        for attr in attrs_to_remove:
            if hasattr(mlp_layer, attr):
                delattr(mlp_layer, attr)
    
    print("Model cleaning complete!")

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

def load_glotcc_data(language_code, tokenizer, target_tokens=1000000, chunk_size=512):
    """
    Load text data from GlotCC-V1 dataset for the target language and collect target number of tokens.
    Split documents into chunks of specified size for training only.
    """
    print(f"Loading GlotCC-V1 training data for language: {language_code}")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Chunk size: {chunk_size} tokens")
    
    # Load the dataset for the specific language
    dataset_name = "cis-lmu/GlotCC-V1"
    
    # Load dataset in streaming mode for memory efficiency
    language_code = flores_lang_mapping.get(language_code, language_code)
    language_code = language_code.replace('_', '-')
    dataset = load_dataset(dataset_name, language_code, split='train', streaming=True)

    all_chunks = []
    total_tokens_collected = 0
    documents_processed = 0
    
    print("Processing documents and collecting tokens...")
    
    for example in tqdm(dataset, desc="Processing documents"):
        if total_tokens_collected >= target_tokens:
            break
            
        # Extract text content 
        text = example.get('text', '') or example.get('content', '') or str(example)
        
        if not text or len(text.strip()) < 50:  # Skip very short texts
            continue
            
        # Tokenize the full document
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Skip documents that are too short
        if len(tokens) < chunk_size // 2:
            continue
        
        # Split tokens into chunks
        for i in range(0, len(tokens), chunk_size):
            if total_tokens_collected >= target_tokens:
                break
                
            chunk_tokens = tokens[i:i + chunk_size]
            
            # Only keep chunks that are at least half the target size
            if len(chunk_tokens) >= chunk_size // 2:
                # Decode back to text
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                all_chunks.append(chunk_text)
                total_tokens_collected += len(chunk_tokens)
        
        documents_processed += 1
        
        # Print progress every 100 documents
        if documents_processed % 100 == 0:
            print(f"Processed {documents_processed} documents, collected {total_tokens_collected:,} tokens in {len(all_chunks)} chunks")
    
    print(f"Completed: Processed {documents_processed} documents")
    print(f"Collected {total_tokens_collected:,} tokens in {len(all_chunks)} chunks")
    print(f"Average tokens per chunk: {total_tokens_collected / len(all_chunks):.1f}")
    
    if len(all_chunks) == 0:
        raise ValueError(f"No valid text chunks found for language '{language_code}'")
    
    return all_chunks

def load_flores_validation_data(language_code, tokenizer, max_samples=1000, chunk_size=512):
    """
    Load validation data from FLORES-200 dataset for the target language.
    """
    print(f"Loading FLORES-200 validation data for language: {language_code}")
    
    # Convert language code if needed
    flores_lang_code = flores_lang_mapping.get(language_code, language_code)
    if flores_lang_code=="ekk_Latn":
        flores_lang_code="est_Latn"
    
    try:    
        # Load FLORES-200 devtest split (which is commonly used for validation)
        dataset = load_dataset("facebook/flores", name="all", split="dev")
        flores_column_name = f"sentence_{flores_lang_code}"
        
        # Filter for the target language
        language_data = []
        for example in dataset:
            text = example[flores_column_name]
            if text and len(text.strip()) > 10:  # Skip very short texts
                language_data.append(text)
        if not language_data:
            print(f"Warning: No data found for language code '{flores_lang_code}' in FLORES-200")
            print(f"Available languages: {list(dataset[0].keys())}")
            # Fallback: try the original language code
            if language_code != flores_lang_code:
                print(f"Trying original language code: {language_code}")
                for example in dataset:
                    if language_code in example:
                        text = example[language_code]
                        if text and len(text.strip()) > 10:
                            language_data.append(text)
        
        if not language_data:
            raise ValueError(f"No validation data found for language '{language_code}' or '{flores_lang_code}' in FLORES-200")
        
        # Limit to max_samples
        if len(language_data) > max_samples:
            language_data = language_data[:max_samples]
        
        print(f"Loaded {len(language_data)} validation sentences from FLORES-200")
        
        # Process the sentences (they're typically short, so we might combine some)
        validation_chunks = []
        current_chunk = ""
        
        for sentence in language_data:
            # Add sentence to current chunk
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Check if adding this sentence would exceed chunk_size
            tokens = tokenizer.encode(potential_chunk, add_special_tokens=False)
            
            if len(tokens) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start a new one
                if current_chunk:
                    validation_chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            validation_chunks.append(current_chunk)
        
        # Calculate validation token count
        val_tokens = sum(len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in validation_chunks)
        
        print(f"Created {len(validation_chunks)} validation chunks from FLORES-200")
        print(f"Validation data contains {val_tokens:,} tokens")
        print(f"Average tokens per validation chunk: {val_tokens / len(validation_chunks):.1f}")
        
        return validation_chunks
        
    except Exception as e:
        print(f"Error loading FLORES-200 data: {e}")
        print("Falling back to using a small portion of training data for validation...")
        return None

def validate_model(model, dataloader, device=None):
    """Validate the model on validation dataset"""
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def fine_tune_language_neurons(model, tokenizer, activation_masks, lang_to_idx, 
                                   target_lang, target_tokens=1000000, epochs=3, learning_rate=1e-4,
                                   batch_size=4, chunk_size=512, 
                                   validate_every=500, output_dir="results/fine_tuning",
                                   fine_tune_mode="random", random_seed=42):
    
    print(f"Starting fine-tuning for language: {target_lang}")
    print(f"Fine-tuning mode: {fine_tune_mode}")
    print(f"Validation every {validate_every} steps")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data from GlotCC-V1
    train_chunks = load_glotcc_data(target_lang, tokenizer, target_tokens, chunk_size)
    
    # Load validation data from FLORES-200
    val_chunks = load_flores_validation_data(target_lang, tokenizer, max_samples=1000, chunk_size=chunk_size)
    
    # Create datasets and dataloaders
    train_dataset = LanguageSpecificDataset(train_chunks, tokenizer, chunk_size)
    val_dataset = LanguageSpecificDataset(val_chunks, tokenizer, chunk_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize appropriate fine-tuner based on mode
    if fine_tune_mode == "random":
        fine_tuner = RandomNeuronFineTuner(model, tokenizer, activation_masks, target_lang, lang_to_idx, seed=random_seed)
        fine_tuner.install_hooks()
    else:
        raise ValueError(f"Unknown fine-tuning mode: {fine_tune_mode}. Choose from 'language', 'random', 'full'")
    
    # Setup optimizer with the appropriate parameters
    if fine_tune_mode == "full":
        # For full model fine-tuning, use a smaller learning rate
        actual_lr = learning_rate * 0.1  # Scale down for full model
        print(f"Using scaled learning rate {actual_lr} for full model fine-tuning")
    else:
        actual_lr = learning_rate
    
    optimizer = optim.AdamW(fine_tuner.target_params, lr=actual_lr, weight_decay=0.01)
    
    # Debug: Count parameters by component
    print("\nParameter breakdown:")
    total_trainable = sum(p.numel() for p in fine_tuner.target_params)
    total_model_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Trainable parameters: {total_trainable:,}")
    print(f"  Total model parameters: {total_model_params:,}")
    print(f"  Percentage of model being trained: {(total_trainable/total_model_params)*100:.4f}%")
    
    # Training loop with step-based validation
    training_losses = []
    validation_losses = []
    step_numbers = []  # Track step numbers for validation
    best_val_loss = float('inf')
    best_step = 0
    global_step = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{epochs} - Training")
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            global_step += 1
            
            # Move batch to device - get device from model
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(fine_tuner.target_params, max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Step-based validation (best model saving only)
            if global_step % validate_every == 0:
                print(f"\nStep {global_step} - Validation")
                val_loss = validate_model(model, val_dataloader)
                
                # Store metrics
                step_numbers.append(global_step)
                validation_losses.append(val_loss)
                avg_train_loss = epoch_loss / num_batches
                training_losses.append(avg_train_loss)
                
                print(f"Step {global_step} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save ONLY if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = global_step
                    
                    print(f"New best validation loss: {val_loss:.4f} - Saving model...")
                    
                    # 1. Remove custom forward methods
                    fine_tuner.remove_hooks()
                    
                    # 2. Integrate trained parameters into model weights
                    integrate_trained_parameters(model, fine_tuner)
                    
                    # 3. Clean model of custom attributes
                    clean_model_for_saving(model, fine_tuner)
                    
                    # 4. Save the clean model (overwrite previous best)
                    best_model_dir = os.path.join(output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                    
                    # 5. Recreate the fine-tuner for continued training
                    if fine_tune_mode != "full":
                        if fine_tune_mode == "random":
                            fine_tuner = RandomNeuronFineTuner(model, tokenizer, activation_masks, target_lang, lang_to_idx, seed=random_seed)
                        fine_tuner.install_hooks()
                        
                        # 6. Recreate optimizer with new parameters
                        optimizer = optim.AdamW(fine_tuner.target_params, lr=actual_lr, weight_decay=0.01)
                    else:
                        # For full model, just reinstall hooks (which do nothing)
                        fine_tuner.install_hooks()
                    
                    print(f"Best model saved to {best_model_dir}")
                else:
                    print(f"No improvement (best: {best_val_loss:.4f} at step {best_step})")
                
                # Return to training mode
                model.train()
        
        # End of epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"End of Epoch {epoch + 1} - Average Train Loss: {avg_epoch_loss:.4f}")
    
    training_info = {
        "target_language": target_lang,
        "fine_tune_mode": fine_tune_mode,
        "random_seed": random_seed if fine_tune_mode == "random" else None,
        "num_train_samples": len(train_chunks),
        "num_val_samples": len(val_chunks),
        "validation_source": "FLORES-200" if val_chunks != train_chunks[:len(val_chunks)] else "training_split",
        "epochs": epochs,
        "total_steps": global_step,
        "validate_every_steps": validate_every,
        "learning_rate": learning_rate,
        "actual_learning_rate": actual_lr,
        "batch_size": batch_size,
        "max_length": chunk_size,
        "step_numbers": step_numbers,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "final_train_loss": training_losses[-1] if training_losses else None,
        "final_val_loss": validation_losses[-1] if validation_losses else None,
        "model_format": "huggingface_pytorch",
        "files_created": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json"],
        "trainable_parameters": sum(p.numel() for p in fine_tuner.target_params),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "training_percentage": (sum(p.numel() for p in fine_tuner.target_params) / sum(p.numel() for p in model.parameters())) * 100
    }
    
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nFine-tuning completed!")
    print(f"Total steps: {global_step}")
    print(f"Validation source: {'FLORES-200' if 'FLORES-200' in training_info['validation_source'] else 'Training data split'}")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
    
    return model, training_losses, validation_losses

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def get_lang_to_idx():
    """Compute average activation values for each language"""
    lang_names = [
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
    # Create a mapping from language name to index
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    return lang_to_idx

def main():
    parser = argparse.ArgumentParser(description="Fine-tune random neurons as baseline comparison")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                       help="Model to fine-tune")
    parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3_5",
                       help="Path to activation masks")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code (e.g., 'es', 'fr', 'de')")
    parser.add_argument("--output_dir", type=str, default="results/random_baseline",
                       help="Output directory for fine-tuning results")
    parser.add_argument("--num_tokens", type=int, default=1000000,
                       help="Number of training tokens from GlotCC-V1")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--evaluate_only", action='store_true',
                       help="Only evaluate existing fine-tuned model")
    parser.add_argument("--validate_every", type=int, default=500,
                       help="Validate and save every N steps (default: 500)")
    parser.add_argument("--use_fp16", action='store_true',
                       help="Use float16 precision instead of model's original precision")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="Specific GPU ID to use (default: 0)")
    
    parser.add_argument("--fine_tune_mode", type=str, default="random", 
                       choices=["language", "random", "full"],
                       help="Fine-tuning mode: 'language' (language-specific neurons), "
                            "'random' (random neurons with same count), 'full' (entire model)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible random neuron selection (default: 42)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Fine-tuning mode: {args.fine_tune_mode}")
    print(f"Target language: {args.target_lang}")
    print(f"Model: {args.model}")
    print(f"Training tokens: {args.num_tokens:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Validate and save every: {args.validate_every} steps")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Validation: FLORES-200 (with fallback to training split)")
    if args.fine_tune_mode == "random":
        print(f"Random seed: {args.random_seed}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check device availability and set device
    if args.gpu_id >= torch.cuda.device_count():
        print(f"Warning: GPU {args.gpu_id} not available. Available GPUs: {torch.cuda.device_count()}")
        device = f"cuda:{min(args.gpu_id, torch.cuda.device_count()-1)}"
    else:
        device = f"cuda:{args.gpu_id}"
    print(f"Using GPU {device}: {torch.cuda.get_device_name(int(device.split(':')[1]))}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(int(device.split(':')[1])).total_memory / 1e9:.1f} GB")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.use_fp16 else None,
        device_map={"": device} 
    )

    # Ensure model is on the correct device
    model = model.to(device)
    
    # Clear GPU cache if using GPU
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    # Load activation masks
    activation_masks = None
    lang_to_idx = None
    if args.fine_tune_mode in ["language", "random"]:
        print("Loading activation masks...")
        activation_masks = torch.load(args.activation_mask)
        lang_to_idx = get_lang_to_idx()
    
    # Fine-tune the model
    model, training_losses, validation_losses = fine_tune_language_neurons(
        model=model,
        tokenizer=tokenizer,
        activation_masks=activation_masks,
        lang_to_idx=lang_to_idx,
        target_lang=args.target_lang,
        target_tokens=args.num_tokens,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        chunk_size=args.max_length,
        validate_every=args.validate_every,
        output_dir=args.output_dir,
        fine_tune_mode=args.fine_tune_mode,
        random_seed=args.random_seed
    )
    
if __name__ == "__main__":
    main()