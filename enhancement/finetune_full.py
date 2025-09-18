import argparse
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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

class FineTuner:
    """Fine-tuner class that can handle both full model and MLP-only fine-tuning"""
    
    def __init__(self, model, tokenizer, fine_tune_mode="mlp_only"):
        self.model = model
        self.tokenizer = tokenizer
        self.fine_tune_mode = fine_tune_mode
        
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters based on mode
        if fine_tune_mode == "mlp_only":
            self.unfreeze_mlp_parameters()
        elif fine_tune_mode == "full_model":
            self.unfreeze_all_parameters()
        else:
            raise ValueError(f"Unknown fine-tuning mode: {fine_tune_mode}. Choose 'mlp_only' or 'full_model'")
        
    def unfreeze_mlp_parameters(self):
        """Unfreeze only MLP parameters across all layers"""
        print("Unfreezing MLP parameters only...")
        
        unfrozen_params = 0
        self.target_params = []
        
        # Count total layers
        num_layers = len(self.model.model.layers)
        print(f"Processing {num_layers} transformer layers...")
        
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]
            mlp_layer = layer.mlp
            
            # Unfreeze all MLP parameters
            mlp_params = []
            
            # Gate projection (or gate_up combined)
            if hasattr(mlp_layer, 'gate_up_proj'):
                # Combined gate_up_proj architecture (e.g., some LLaMA variants)
                mlp_layer.gate_up_proj.weight.requires_grad = True
                if mlp_layer.gate_up_proj.bias is not None:
                    mlp_layer.gate_up_proj.bias.requires_grad = True
                mlp_params.extend([p for p in mlp_layer.gate_up_proj.parameters() if p.requires_grad])
                
            elif hasattr(mlp_layer, 'gate_proj') and hasattr(mlp_layer, 'up_proj'):
                # Separate gate_proj and up_proj architecture
                mlp_layer.gate_proj.weight.requires_grad = True
                if mlp_layer.gate_proj.bias is not None:
                    mlp_layer.gate_proj.bias.requires_grad = True
                    
                mlp_layer.up_proj.weight.requires_grad = True
                if mlp_layer.up_proj.bias is not None:
                    mlp_layer.up_proj.bias.requires_grad = True
                    
                mlp_params.extend([p for p in mlp_layer.gate_proj.parameters() if p.requires_grad])
                mlp_params.extend([p for p in mlp_layer.up_proj.parameters() if p.requires_grad])
                
            else:
                raise ValueError(f"Unknown MLP architecture at layer {layer_idx}. "
                               f"Expected 'gate_up_proj' or ('gate_proj' and 'up_proj'), "
                               f"but found: {list(mlp_layer._modules.keys())}")
            
            # Down projection
            mlp_layer.down_proj.weight.requires_grad = True
            if mlp_layer.down_proj.bias is not None:
                mlp_layer.down_proj.bias.requires_grad = True
            mlp_params.extend([p for p in mlp_layer.down_proj.parameters() if p.requires_grad])
            
            # Add to target params
            self.target_params.extend(mlp_params)
            layer_param_count = sum(p.numel() for p in mlp_params)
            unfrozen_params += layer_param_count
            
            if layer_idx % 8 == 0 or layer_idx == num_layers - 1:  # Print every 8 layers
                print(f"  Layer {layer_idx}: {layer_param_count:,} MLP parameters unfrozen")
        
        print(f"Total unfrozen MLP parameters: {unfrozen_params:,}")
        print(f"MLP parameters per layer (avg): {unfrozen_params // num_layers:,}")
        
        return unfrozen_params
    
    def unfreeze_all_parameters(self):
        """Unfreeze all model parameters for full model fine-tuning"""
        print("Unfreezing ALL model parameters...")
        
        unfrozen_params = 0
        self.target_params = []
        
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            self.target_params.append(param)
            unfrozen_params += param.numel()
        
        print(f"Total unfrozen parameters: {unfrozen_params:,}")
        
        # Print breakdown by component type
        component_counts = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'embed' in name.lower():
                    component_type = 'embeddings'
                elif 'mlp' in name.lower():
                    component_type = 'mlp'
                elif 'self_attn' in name.lower() or 'attention' in name.lower():
                    component_type = 'attention'
                elif 'norm' in name.lower() or 'ln' in name.lower():
                    component_type = 'layer_norm'
                elif 'lm_head' in name.lower():
                    component_type = 'lm_head'
                else:
                    component_type = 'other'
                
                if component_type not in component_counts:
                    component_counts[component_type] = 0
                component_counts[component_type] += param.numel()
        
        print("\nParameter breakdown by component:")
        for component, count in sorted(component_counts.items()):
            percentage = (count / unfrozen_params) * 100
            print(f"  {component}: {count:,} parameters ({percentage:.2f}%)")
        
        return unfrozen_params
    
    def install_hooks(self):
        """No custom hooks needed for this implementation"""
        pass
    
    def remove_hooks(self):
        """No hooks to remove for this implementation"""
        pass

# Language code mapping for FLORES-200
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

def validate_model(model, dataloader, use_multi_gpu=False):
    """Validate the model on validation dataset with multi-GPU support"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if use_multi_gpu:
                # For multi-GPU, get device from first parameter
                model_device = next(model.parameters()).device
                input_ids = batch['input_ids'].to(model_device)
                attention_mask = batch['attention_mask'].to(model_device)
                labels = batch['labels'].to(model_device)
            else:
                # For single GPU, get device from model
                device = next(model.parameters()).device
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

def fine_tune_model(model, tokenizer, target_lang, fine_tune_mode="mlp_only", target_tokens=1000000, 
                    epochs=3, learning_rate=1e-5, batch_size=4, chunk_size=512, validate_every=500, 
                    output_dir="results", use_multi_gpu=False):
    """
    Fine-tune the model with flexible parameter selection and multi-GPU support.
    
    Args:
        fine_tune_mode: "mlp_only" or "full_model"
        use_multi_gpu: Whether multi-GPU is being used (for device handling)
        validate_every: Validate every N steps (default: 500)
        Other args: same as before
    """
    
    print(f"Starting fine-tuning for language: {target_lang}")
    print(f"Fine-tuning mode: {fine_tune_mode}")
    print(f"Multi-GPU: {use_multi_gpu}")
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

    # Initialize fine-tuner
    fine_tuner = FineTuner(model, tokenizer, fine_tune_mode)
    fine_tuner.install_hooks()  # No-op for this implementation
    
    # Setup optimizer
    optimizer = optim.AdamW(fine_tuner.target_params, lr=learning_rate, weight_decay=0.01)
    
    # Debug: Count parameters by component
    print("\nParameter breakdown:")
    total_trainable = sum(p.numel() for p in fine_tuner.target_params)
    total_model_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Trainable parameters: {total_trainable:,}")
    print(f"  Total model parameters: {total_model_params:,}")
    print(f"  Percentage of model being trained: {(total_trainable/total_model_params)*100:.4f}%")
    
    # Estimate memory usage
    param_memory_gb = (total_trainable * 4) / (1024**3)  # 4 bytes per float32 parameter
    grad_memory_gb = param_memory_gb  # Gradients take same space as parameters
    print(f"  Estimated memory for parameters: {param_memory_gb:.2f} GB")
    print(f"  Estimated memory for gradients: {grad_memory_gb:.2f} GB")
    print(f"  Total estimated training memory: {(param_memory_gb + grad_memory_gb):.2f} GB")
    
    # Training loop with step-based validation
    training_losses = []
    validation_losses = []
    step_numbers = []  # Track step numbers for validation
    best_val_loss = float('inf')
    best_step = 0
    global_step = 0
    
    # Determine target device for data movement
    if use_multi_gpu:
        # For multi-GPU, use device of the first parameter
        target_device = next(model.parameters()).device
    else:
        # For single GPU, use the model's device
        target_device = next(model.parameters()).device
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{epochs} - Training")
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            global_step += 1
            
            # Move batch to appropriate device
            input_ids = batch['input_ids'].to(target_device)
            attention_mask = batch['attention_mask'].to(target_device)
            labels = batch['labels'].to(target_device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping - adjust based on fine-tuning mode
            max_norm = 0.5 if fine_tune_mode == "full_model" else 1.0
            torch.nn.utils.clip_grad_norm_(fine_tuner.target_params, max_norm=max_norm)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Step-based validation (best model saving only)
            if global_step % validate_every == 0:
                print(f"\nStep {global_step} - Validation")
                val_loss = validate_model(model, val_dataloader, use_multi_gpu)
                
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
                    
                    best_model_dir = os.path.join(output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)
                    
                    # Save model with appropriate serialization
                    if use_multi_gpu:
                        model.save_pretrained(best_model_dir, safe_serialization=False)
                    else:
                        model.save_pretrained(best_model_dir)
                    
                    tokenizer.save_pretrained(best_model_dir)
                    
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
        "multi_gpu": use_multi_gpu,
        "num_train_samples": len(train_chunks),
        "num_val_samples": len(val_chunks),
        "validation_source": "FLORES-200" if val_chunks != train_chunks[:len(val_chunks)] else "training_split",
        "epochs": epochs,
        "total_steps": global_step,
        "validate_every_steps": validate_every,
        "learning_rate": learning_rate,
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
        "training_percentage": (sum(p.numel() for p in fine_tuner.target_params) / sum(p.numel() for p in model.parameters())) * 100,
        "mlp_only": fine_tune_mode == "mlp_only",
        "full_model": fine_tune_mode == "full_model"
    }
    
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nFine-tuning completed!")
    print(f"Total steps: {global_step}")
    print(f"Mode: {fine_tune_mode}")
    print(f"Multi-GPU: {use_multi_gpu}")
    print(f"Validation source: {'FLORES-200' if 'FLORES-200' in training_info['validation_source'] else 'Training data split'}")
    print(f"Trained {training_info['trainable_parameters']:,} parameters ({training_info['training_percentage']:.4f}% of model)")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
    
    return model, training_losses, validation_losses

def main():
    parser = argparse.ArgumentParser(description="Fine-tune language models with flexible parameter selection")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                       help="Model to fine-tune")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code (e.g., 'es', 'fr', 'de')")
    parser.add_argument("--fine_tune_mode", type=str, choices=["mlp_only", "full_model"], 
                       default="mlp_only",
                       help="Fine-tuning mode: 'mlp_only' (default) or 'full_model'")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for fine-tuning results (auto-generated if not provided)")
    parser.add_argument("--num_tokens", type=int, default=1000000,
                       help="Number of training tokens from GlotCC-V1")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--validate_every", type=int, default=500,
                       help="Validate and save every N steps (default: 500)")
    parser.add_argument("--use_fp16", action='store_true',
                       help="Use float16 precision to save memory")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="Specific GPU ID to use for single GPU mode (default: 0)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of steps to accumulate gradients (default: 1)")
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"results/{args.fine_tune_mode}_{args.target_lang}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if we should use multi-GPU (automatic for full_model)
    use_multi_gpu = args.fine_tune_mode == "full_model"
    
    print(f"Flexible Fine-tuning Script")
    print(f"Fine-tuning mode: {args.fine_tune_mode}")
    print(f"Multi-GPU (automatic for full_model): {use_multi_gpu}")
    print(f"Target language: {args.target_lang}")
    print(f"Model: {args.model}")
    print(f"Training tokens: {args.num_tokens:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Validate and save every: {args.validate_every} steps")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation: FLORES-200 (with fallback to training split)")
    print(f"FP16: {args.use_fp16}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading with conditional device mapping
    if use_multi_gpu:
        print("Using multi-GPU setup with device_map='auto'")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if args.use_fp16 else None,
            device_map="auto"  # Automatically distribute across GPUs
        )
        
        # Print device mapping
        print("\nModel device mapping:")
        if hasattr(model, 'hf_device_map'):
            for module_name, device in model.hf_device_map.items():
                print(f"  {module_name}: {device}")
        
        # Clear GPU cache on all devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        
    else:
        # Single GPU setup
        if args.gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu_id} not available. Available GPUs: {torch.cuda.device_count()}")
            device = f"cuda:{min(args.gpu_id, torch.cuda.device_count()-1)}"
        else:
            device = f"cuda:{args.gpu_id}"
        
        print(f"Using single GPU {device}: {torch.cuda.get_device_name(int(device.split(':')[1]))}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(int(device.split(':')[1])).total_memory / 1e9:.1f} GB")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if args.use_fp16 else None,
            device_map={"": device}
        )
        model = model.to(device)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Fine-tune the model
    model, training_losses, validation_losses = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        target_lang=args.target_lang,
        fine_tune_mode=args.fine_tune_mode,
        target_tokens=args.num_tokens,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        chunk_size=args.max_length,
        validate_every=args.validate_every,
        output_dir=args.output_dir,
        use_multi_gpu=use_multi_gpu
    )

if __name__ == "__main__":
    main()