import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
parser.add_argument("--finetuned_model", type=str, default="finetune/Llama-3.1-8B_5_is_100M/best_model")
parser.add_argument("--activation_mask", type=str, default="activation_mask/llama-3.1-5")
parser.add_argument("--target_lang", type=str, default="is")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="weight_analysis_plots")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def compute_average_activations():
    """Compute average activation values for each language.
    """
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
    
    return lang_to_idx, lang_names

def load_target_neuron_indices(activation_mask_path, target_lang):
    """
    Loads the pre-computed neuron indices for the specified target language
    from a single activation mask file.
    """
    # Get the mapping from language name to index
    lang_to_idx, _ = compute_average_activations()
    
    if target_lang not in lang_to_idx:
        raise ValueError(f"Target language '{target_lang}' not found in the predefined list.")
    
    # Load the main activation masks file
    try:
        activation_masks = torch.load(activation_mask_path)
    except FileNotFoundError:
        print(f"Error: Activation mask file not found at {activation_mask_path}")
        return None
        
    # Get the neuron indices for the specific target language
    lang_index = lang_to_idx[target_lang]
        
    # The selected object is a list of tensors, where each tensor corresponds to a layer.
    # We convert it to a dictionary mapping layer index to the tensor.
    raw_indices_list = activation_masks[lang_index]
    
    # Create the dictionary from the list of tensors
    target_neuron_indices = {i: tensor for i, tensor in enumerate(raw_indices_list)}
    
    print(f"Loaded neuron indices for language '{target_lang}'.")
    return target_neuron_indices

def get_weight_diffs(base_model, finetuned_model, target_neuron_indices):
    """
    Calculates the weight differences for the target-specific MLP layers.
    """
    weight_diffs = {}
    
    for layer_idx, indices in target_neuron_indices.items():
            
        base_mlp = base_model.model.layers[layer_idx].mlp
        finetuned_mlp = finetuned_model.model.layers[layer_idx].mlp
        
        layer_diffs = {}
        
        # Check for Llama architecture (gate_proj, up_proj, down_proj)
        if hasattr(base_mlp, 'gate_proj') and hasattr(base_mlp, 'up_proj') and hasattr(base_mlp, 'down_proj'):
            # Gate Proj
            base_gate_w = base_mlp.gate_proj.weight.data
            finetuned_gate_w = finetuned_mlp.gate_proj.weight.data
            diff_gate = finetuned_gate_w[indices, :] - base_gate_w[indices, :]
            layer_diffs['gate_proj'] = diff_gate
            
            # Up Proj
            base_up_w = base_mlp.up_proj.weight.data
            finetuned_up_w = finetuned_mlp.up_proj.weight.data
            diff_up = finetuned_up_w[indices, :] - base_up_w[indices, :]
            layer_diffs['up_proj'] = diff_up
            
            # Down Proj
            base_down_w = base_mlp.down_proj.weight.data
            finetuned_down_w = finetuned_mlp.down_proj.weight.data
            diff_down = finetuned_down_w[:, indices] - base_down_w[:, indices]
            layer_diffs['down_proj'] = diff_down
        
        elif hasattr(base_mlp, 'gate_up_proj') and hasattr(base_mlp, 'down_proj'):
            # Combined gate_up_proj architecture (like Llama-3.1)
            base_gate_up_w = base_mlp.gate_up_proj.weight.data
            finetuned_gate_up_w = finetuned_mlp.gate_up_proj.weight.data
            
            # Extract gate part
            hidden_size = base_gate_up_w.shape[0] // 2
            gate_indices = indices
            diff_gate = (finetuned_gate_up_w[gate_indices, :] - base_gate_up_w[gate_indices, :])
            
            # Extract up part
            up_indices = indices + hidden_size
            diff_up = (finetuned_gate_up_w[up_indices, :] - base_gate_up_w[up_indices, :])
            
            layer_diffs['gate_proj'] = diff_gate
            layer_diffs['up_proj'] = diff_up
            
            # Down Proj
            base_down_w = base_mlp.down_proj.weight.data
            finetuned_down_w = finetuned_mlp.down_proj.weight.data
            diff_down = finetuned_down_w[:, indices] - base_down_w[:, indices]
            layer_diffs['down_proj'] = diff_down

        weight_diffs[layer_idx] = layer_diffs
        
    return weight_diffs

def plot_and_save_hist(weight_diffs, layer_idx, proj_name, output_dir):
    """
    Plots a histogram of the weight differences and saves it to a file.
    """
    if layer_idx in weight_diffs and proj_name in weight_diffs[layer_idx]:
        diff_tensor = weight_diffs[layer_idx][proj_name].cpu().numpy().flatten()
        if diff_tensor.size == 0:
            print(f"Warning: No data to plot for Layer {layer_idx}, {proj_name}.")
            return
        
        plt.figure(figsize=(5, 4))
        plt.hist(diff_tensor, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.title(f"Weight Difference Distribution: Layer {layer_idx}, {proj_name}", fontsize=13)
        plt.xlabel("Weight Change", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        filename = f"hist_layer_{layer_idx}_{proj_name}"
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), format='pdf')
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.close()
        print(f"Saved histogram to {filename}")


def plot_and_save_magnitude(weight_diffs, output_dir):
    """
    Plots the average magnitude of weight changes per layer and projection and saves it.
    """
    layers = sorted(weight_diffs.keys())
    projections = ['gate_proj', 'up_proj', 'down_proj']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']  # Circle, Square, Triangle
    
    avg_magnitudes = {proj: [] for proj in projections}
    
    for layer_idx in layers:
        for proj in projections:
            if proj in weight_diffs[layer_idx]:
                magnitude = torch.norm(weight_diffs[layer_idx][proj])
                avg_magnitudes[proj].append(magnitude.item())
            else:
                avg_magnitudes[proj].append(0)
                
    plt.figure(figsize=(5, 4))
    for i, proj in enumerate(projections):
        plt.plot(layers, avg_magnitudes[proj], marker=markers[i], markersize=4, 
                color=colors[i], linewidth=2, label=proj.replace('_', ' ').title())
    
    plt.title("L2-Norm of Weight Differences Across Layers", fontsize=13)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("L2 Norm of Weight Differences", fontsize=12)
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, 
              title='Projections', title_fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the figure
    filename = "l2_norm_across_layers"
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), format='pdf')
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close()
    print(f"Saved L2 norm plot to {filename}")

def plot_and_save_weight_change_scatter(base_model, finetuned_model, target_neuron_indices, weight_diffs, output_dir):
    """
    Generates scatter plots of original weight vs. change for each layer and saves them.
    """
    colors = {'gate_proj': '#1f77b4', 'up_proj': '#ff7f0e', 'down_proj': '#2ca02c'}
    
    for layer_idx, indices in target_neuron_indices.items():
        base_mlp = base_model.model.layers[layer_idx].mlp
        
        for proj_name, diff_tensor in weight_diffs[layer_idx].items():
            if proj_name == 'down_proj':
                if hasattr(base_mlp, 'gate_up_proj'):
                    # For combined architecture, get from down_proj
                    base_weights = base_mlp.down_proj.weight.data[:, indices].cpu().numpy().flatten()
                else:
                    base_weights = base_mlp.down_proj.weight.data[:, indices].cpu().numpy().flatten()
            elif proj_name == 'gate_proj':
                if hasattr(base_mlp, 'gate_up_proj'):
                    # For combined architecture, get gate part
                    base_weights = base_mlp.gate_up_proj.weight.data[indices, :].cpu().numpy().flatten()
                else:
                    base_weights = base_mlp.gate_proj.weight.data[indices, :].cpu().numpy().flatten()
            elif proj_name == 'up_proj':
                if hasattr(base_mlp, 'gate_up_proj'):
                    # For combined architecture, get up part
                    hidden_size = base_mlp.gate_up_proj.weight.data.shape[0] // 2
                    up_indices = indices + hidden_size
                    base_weights = base_mlp.gate_up_proj.weight.data[up_indices, :].cpu().numpy().flatten()
                else:
                    base_weights = base_mlp.up_proj.weight.data[indices, :].cpu().numpy().flatten()
            
            diffs = diff_tensor.cpu().numpy().flatten()
            
            if base_weights.size > 0:
                plt.figure(figsize=(5, 4))
                plt.scatter(base_weights, diffs, alpha=0.6, s=2, 
                           color=colors.get(proj_name, '#1f77b4'))
                plt.title(f"Weight Change vs. Original Weight: Layer {layer_idx}, {proj_name.replace('_', ' ').title()}", 
                         fontsize=13)
                plt.xlabel("Original Weight Value", fontsize=12)
                plt.ylabel("Weight Change (Δw)", fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                filename = f"scatter_layer_{layer_idx}_{proj_name}"
                plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), format='pdf')
                plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
                plt.close()
                print(f"Saved scatter plot to {filename}")

def plot_and_save_all_projections_violinplot(weight_diffs, output_dir):
    """
    Generates violin plots comparing the distribution of weight changes across layers for each projection.
    """
    projections = ['gate_proj', 'up_proj', 'down_proj']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Get all layer indices to maintain consistent x-axis
    all_layer_indices = sorted(weight_diffs.keys())
    
    for i, proj_name in enumerate(projections):
        data_to_plot = []
        positions = []
        layer_labels = []
        
        for pos, layer_idx in enumerate(all_layer_indices):
            layer_labels.append(str(layer_idx))
            
            if (layer_idx in weight_diffs and 
                proj_name in weight_diffs[layer_idx]):
                # Get the flattened data
                flat_data = weight_diffs[layer_idx][proj_name].cpu().numpy().flatten()
                # Only add data if the array is not empty
                if flat_data.size > 0:
                    data_to_plot.append(flat_data)
                    positions.append(pos + 1)  # positions start from 1
                else:
                    print(f"Warning: Empty data for layer {layer_idx}, {proj_name}")
            # If layer doesn't have this projection or is empty, we skip adding data
            # but still keep the position in layer_labels for x-axis continuity
        
        if len(data_to_plot) == 0:
            print(f"Warning: No data to plot for {proj_name} violin plot.")
            continue
            
        plt.figure(figsize=(5, 4))
        violin_parts = plt.violinplot(data_to_plot, positions=positions, 
                                     showmeans=True, showmedians=True)
        
        # Color the violin plots
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Style the other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in violin_parts:
                violin_parts[partname].set_edgecolor('black')
                violin_parts[partname].set_linewidth(0.8)
        
        # Set x-axis labels for ALL layers (including empty ones)
        plt.xticks(range(1, len(layer_labels) + 1), layer_labels)
        
        # Show only every second x-tick label to reduce clutter
        ax = plt.gca()
        tick_labels = ax.get_xticklabels()
        for j, label in enumerate(tick_labels):
            if j % 2 != 0:  # Hide odd-indexed labels (keep every second one)
                label.set_visible(False)
        
        plt.title(f"Weight Change Distribution: {proj_name.replace('_', ' ').title()}", fontsize=13)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Weight Change", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f"violinplot_{proj_name}"
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), format='pdf')
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.close()
        print(f"Saved violin plot to {filename}") 

def plot_and_save_all_projections_boxplot(weight_diffs, output_dir):
    """
    Generates box plots comparing the distribution of weight changes across layers for each projection.
    """
    projections = ['gate_proj', 'up_proj', 'down_proj']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, proj_name in enumerate(projections):
        data_to_plot = []
        layer_labels = []
        
        for layer_idx, layer_diffs in weight_diffs.items():
            if proj_name in layer_diffs:
                # Add data only for layers that have a valid weight difference tensor
                data_to_plot.append(layer_diffs[proj_name].cpu().numpy().flatten())
                layer_labels.append(str(layer_idx))
        
        if len(data_to_plot) == 0:
            print(f"Warning: No data to plot for {proj_name} box plot.")
            continue
            
        plt.figure(figsize=(5, 4))
        box_plot = plt.boxplot(data_to_plot, labels=layer_labels, patch_artist=True,
                              flierprops=dict(marker='o', markersize=2, alpha=0.6))  # Smaller outlier circles
        
        # Color the boxes
        for patch in box_plot['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        
        # Show only every second x-tick label to reduce clutter
        ax = plt.gca()
        tick_labels = ax.get_xticklabels()
        for j, label in enumerate(tick_labels):
            if j % 2 != 0:  # Hide odd-indexed labels (keep every second one)
                label.set_visible(False)
        
        plt.title(f"Weight Change Distribution: {proj_name.replace('_', ' ').title()}", fontsize=13)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Weight Change", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f"boxplot_{proj_name}"
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), format='pdf')
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.close()
        print(f"Saved box plot to {filename}") 
           
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
        
    # Load models
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    print("Loading fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(args.finetuned_model).to(device)
    
    # Load target neuron indices using the new logic
    print("Loading target neuron indices...")
    target_neuron_indices = load_target_neuron_indices(args.activation_mask, args.target_lang)
    if not target_neuron_indices:
        print("Error: Could not load target neuron indices. Exiting.")
        return
        
    # Calculate weight differences
    print("Calculating weight differences...")
    weight_diffs = get_weight_diffs(base_model, finetuned_model, target_neuron_indices)
    
    # Generate and save new scatter plots and violin plots
    print("Generating violin plots...")
    plot_and_save_all_projections_violinplot(weight_diffs, args.output_dir)
    plot_and_save_all_projections_boxplot(weight_diffs, args.output_dir)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()