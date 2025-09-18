import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="activation_mask/aya-8")
parser.add_argument("--output_path", type=str, default="plots")

global args
args = parser.parse_args()

# Simplified style settings to match your snippet
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.3,
    "figure.dpi": 300
})

# Output directory
os.makedirs(args.output_path, exist_ok=True)

# Language codes used (same order as input)
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

# Simplified colors using matplotlib's tab colors
tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
              'steelblue', 'darkseagreen']

# Create language to color mapping
lang_colors = {}
for i, lang in enumerate(languages):
    lang_colors[lang] = tab_colors[i % len(tab_colors)]

def get_language_families():
    """Define language families for grouping"""
    return {
        'Germanic': ['af', 'is'],  # Afrikaans, Icelandic
        'Celtic': ['cy'],  # Welsh
        'Slavic': ['mk', 'sl', 'sk'],  # Macedonian, Slovenian, Slovak
        'Baltic': ['lv', 'lt'],  # Latvian, Lithuanian
        'Uralic': ['et'],  # Estonian
        'Kartvelian': ['ka'],  # Georgian
        'Indo-Aryan': ['ne'],  # Nepali
        'Semitic': ['mt']  # Maltese
    }

def get_non_latin_scripts():
    """Languages with non-Latin scripts that should have asterisks"""
    return {'ka', 'ne', 'mk'}  # Georgian, Nepali, Macedonian (Cyrillic)

def format_language_label(lang):
    """Add asterisk for non-Latin script languages"""
    non_latin = get_non_latin_scripts()
    return f"{lang}*" if lang in non_latin else lang

def order_languages_by_family(languages, lang_families):
    """Order languages by family grouping"""
    ordered = []
    for family, family_langs in lang_families.items():
        for lang in family_langs:
            if lang in languages:
                ordered.append(lang)
    
    # Add any remaining languages not in families
    for lang in languages:
        if lang not in ordered:
            ordered.append(lang)
    
    return ordered

def get_family_positions(ordered_langs, lang_families):
    """Get positions where family boundaries should be drawn"""
    positions = []
    current_pos = 0
    current_family = None
    
    for lang in ordered_langs:
        lang_family = None
        for family, family_langs in lang_families.items():
            if lang in family_langs:
                lang_family = family
                break
        
        if current_family is not None and lang_family != current_family:
            positions.append(current_pos)
        
        current_family = lang_family
        current_pos += 1
    
    return positions

def add_family_separators_and_labels(ax, ordered_langs, lang_families, axis='both'):
    """Add family separators and labels to the plot"""
    positions = get_family_positions(ordered_langs, lang_families)
    
    # Add separator lines
    if axis in ['both', 'x']:
        for pos in positions:
            ax.axvline(x=pos, color='black', linewidth=1.5, alpha=0.7)
    if axis in ['both', 'y']:
        for pos in positions:
            ax.axhline(y=pos, color='black', linewidth=1.5, alpha=0.7)
    
    # Add family labels
    current_pos = 0
    for family, family_langs in lang_families.items():
        family_count = sum(1 for lang in family_langs if lang in ordered_langs)
        if family_count > 0:
            center_pos = current_pos + (family_count - 1) / 2
            
            if axis in ['both', 'x']:
                ax.text(center_pos+0.5, -1.2, family, ha='center', va='center', 
                       fontsize=11, fontweight='bold', rotation=45)
            
            current_pos += family_count

# Load activation mask
print(f"Loading activation mask from: {args.input_path}")
final_indice = torch.load(f"{args.input_path}")

num_languages = len(final_indice)
num_layers = len(final_indice[0])

print(f"Loaded data for {num_languages} languages and {num_layers} layers")

# Build sets of (layer, neuron) pairs per language
lang_neuron_sets = []
for lang_index in range(num_languages):
    neuron_set = set()
    for layer, heads in enumerate(final_indice[lang_index]):
        for head in heads.tolist():
            neuron_set.add((layer, head))
    lang_neuron_sets.append(neuron_set)

print("Built neuron sets for all languages")

# Get language families and order languages
lang_families = get_language_families()
ordered_languages = order_languages_by_family(languages, lang_families)
ordered_indices = [languages.index(lang) for lang in ordered_languages]

print(f"Language order: {ordered_languages}")

# === Plot 1: Family-Grouped Overlap Heatmap ===
print("Creating family-grouped overlap heatmap...")

# Create overlap matrix with ordered languages
overlap_matrix = np.zeros((num_languages, num_languages), dtype=int)
for i, lang_i_idx in enumerate(ordered_indices):
    for j, lang_j_idx in enumerate(ordered_indices):
        intersection = len(lang_neuron_sets[lang_i_idx] & lang_neuron_sets[lang_j_idx])
        overlap_matrix[i, j] = intersection

# Create formatted labels with asterisks
formatted_labels = [format_language_label(lang) for lang in ordered_languages]

plt.figure(figsize=(8, 6))  # Smaller, more compact size
sns.heatmap(
    overlap_matrix,
    xticklabels=formatted_labels,
    yticklabels=formatted_labels,
    cmap="Oranges",
    annot=True,
    fmt="d",
    cbar=True,
    linewidths=0.5,
    annot_kws={"size": 10},
    square=True
)

# Add family separators and labels
ax = plt.gca()
add_family_separators_and_labels(ax, ordered_languages, lang_families, 'both')

# Simplified colorbar styling
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel("Neuron Overlap Count", fontsize=12)

plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)
# plt.title("Language Neuron Overlap by Family", fontsize=13)
plt.tight_layout()
plt.savefig(f"{args.output_path}/language_overlap_family_grouped.png", dpi=300)
plt.savefig(f"{args.output_path}/language_overlap_family_grouped.pdf", format='pdf')
plt.close()

# === Plot 2: Cumulative Distribution Across All Languages ===
print("Creating cumulative distribution plot...")

# Calculate cumulative neuron counts across all languages
layer_counts_all = np.zeros(num_layers)
for lang_index in range(num_languages):
    for layer, heads in enumerate(final_indice[lang_index]):
        layer_counts_all[layer] += len(heads)

plt.figure(figsize=(5, 4))  # Match your snippet size
bars = plt.bar(range(num_layers), layer_counts_all, color='steelblue', 
               alpha=0.8)

plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Neuron Count", fontsize=12)
plt.title("Cumulative Neuron Distribution", fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{args.output_path}/cumulative_neuron_distribution.png", dpi=300)
plt.savefig(f"{args.output_path}/cumulative_neuron_distribution.pdf", format='pdf')
plt.close()

# === Plot 3: Comparative Line Plot for All Languages ===
print("Creating comparative line plot...")

# Prepare data for line plot
layer_counts_by_lang = np.zeros((num_languages, num_layers))
for lang_index in range(num_languages):
    for layer, heads in enumerate(final_indice[lang_index]):
        layer_counts_by_lang[lang_index, layer] = len(heads)

plt.figure(figsize=(5, 4))  # Match your snippet size

# Plot lines for each language using tab colors and simplified styling
for lang_index, lang in enumerate(languages):
    linestyle = '--' if lang in get_non_latin_scripts() else '-'
    
    plt.plot(range(num_layers), layer_counts_by_lang[lang_index], 
            color=lang_colors[lang], linestyle=linestyle, linewidth=2,
            label=format_language_label(lang))

plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Neuron Count", fontsize=12)
plt.title("Neuron Distribution Across Layers", fontsize=13)
plt.grid(alpha=0.3)

# Simplified legend styling to match your snippet
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=8, 
          title='Languages', title_fontsize=9, loc='upper left', bbox_to_anchor=(0.05, 1))

plt.tight_layout()
plt.savefig(f"{args.output_path}/comparative_line_plot.png", dpi=300)
plt.savefig(f"{args.output_path}/comparative_line_plot.pdf", format='pdf')
plt.close()

# === Plot 4: Small Multiples - Individual Language Distributions ===
print("Creating small multiples plot...")

# Calculate grid dimensions
n_langs = len(languages)
cols = 4  # Smaller grid for cleaner look
rows = (n_langs + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))  # Smaller overall size
if rows == 1:
    axes = axes.reshape(1, -1)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Find global max for consistent y-axis scaling
global_max = 0
for lang_index in range(num_languages):
    neuron_counts = [len(heads) for heads in final_indice[lang_index]]
    global_max = max(global_max, max(neuron_counts))

# Create individual plots
for lang_index, lang in enumerate(languages):
    ax = axes_flat[lang_index]
    neuron_counts = [len(heads) for heads in final_indice[lang_index]]
    
    # Use tab color for each language
    lang_color = lang_colors[lang]
    
    # Create bar plot with simplified styling
    bars = ax.bar(range(num_layers), neuron_counts, color=lang_color, alpha=0.8)
    
    # Simplified formatting
    ax.set_title(format_language_label(lang), fontsize=12, fontweight='bold')
    ax.set_ylim(0, global_max * 1.1)
    ax.set_xticks(range(0, num_layers, max(1, num_layers//5)))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(alpha=0.3)

# Add common axes labels
fig.text(0.5, 0.02, 'Layer Index', ha='center', va='center', fontsize=12, fontweight='bold')
fig.text(0.02, 0.5, 'Neuron Count', ha='center', va='center', 
         rotation=90, fontsize=12, fontweight='bold')

# Hide unused subplots
for i in range(n_langs, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, left=0.08, top=0.93)
plt.savefig(f"{args.output_path}/small_multiples_distribution.png", dpi=300)
plt.savefig(f"{args.output_path}/small_multiples_distribution.pdf", format='pdf')
plt.close()

print("All plots created successfully!")