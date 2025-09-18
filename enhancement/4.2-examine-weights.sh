#!/bin/bash

pip install vllm 
pip install numpy==1.26.4

languages=("mt" "af" "is" "cy" "mk" "lv" "lt" "sl" "sk" "et" "ka" "ne")

BASE_MODEL="mistralai/Mistral-Nemo-Base-2407"
ACTIVATION_MASK="activation_mask/nemo-5"
SEED=42
PYTHON_SCRIPT="examine_weights.py"

for lang in "${languages[@]}"; do
    echo ""
    echo "Processing language: $lang"
    echo "------------------------"
    
    FINETUNED_MODEL="finetune_nemo/Mistral-Nemo-Base-2407_5_${lang}_100M/best_model"
    OUTPUT_DIR="weights_nemo/weight_analysis_plots_${lang}"
    
    python "$PYTHON_SCRIPT" \
        --base_model "$BASE_MODEL" \
        --finetuned_model "$FINETUNED_MODEL" \
        --activation_mask "$ACTIVATION_MASK" \
        --target_lang "$lang" \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR"
    
done
