#!/bin/bash

pip install vllm
pip install -U transformers
pip install -U accelerate
pip install numpy==1.26.4 # --> required by Nemo

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

python data.py