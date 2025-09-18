#!/bin/bash

pip install vllm datasets 
pip install -U accelerate

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

pip install numpy==1.26.4 # vllm needs this all of a sudden now !!!

export VLLM_USE_V1=0

MODEL="mistralai/Mistral-Nemo-Base-2407"
MODEL_NAME=${MODEL#*/}
RATIO=5
ACTIVATION_MASK="activation_mask/nemo-${RATIO}"
LANGUAGE="mt"

python finetune_ablate.py --target_lang $LANGUAGE --num_tokens 100000000 --epochs 1 \
                 --output_dir "finetune_nemo/${MODEL_NAME}_${RATIO}_${LANGUAGE}_100M_random" \
                 --model $MODEL --activation_mask $ACTIVATION_MASK --batch_size 2 \
                 --fine_tune_mode random --random_seed 1