#!/bin/bash

# do not run on V100 or so. on this kind of GPUs, VLLM backs up to using an alternative for flash attention and everything crashes
pip install vllm
# pip install -U transformers accelerate torch
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl # --> required by gemma-3
pip install numpy==1.26.4 # --> required by Nemo

HF_TOKEN=*
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_USE_V1=0

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

languages=("mt" "af" "is" "cy" "mk" "lv" "lt" "sl" "sk" "et" "ka" "ne")

model="mistralai/Mistral-Nemo-Base-2407"

export CUDA_VISIBLE_DEVICES=0

for lang in "${languages[@]}"
do
    echo "Running activation.py for language: $lang"
    python activation.py -m $model -l $lang -s "nemo nemo"
done
