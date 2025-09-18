#!/bin/bash

pip install vllm 
pip install -U accelerate huggingface_hub datasets
pip install numpy==1.26.4

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

# ========= SINGLE EVAL. ==========

python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_mt_100M_full/best_model" \
    --target_lang mlt_Latn \
    --output_dir "finetune/mlt_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_af_100M_full/best_model" \
    --target_lang afr_Latn \
    --output_dir "finetune/afr_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_is_100M_full/best_model" \
    --target_lang isl_Latn \
    --output_dir "finetune/isl_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_cy_100M_full/best_model" \
    --target_lang cym_Latn \
    --output_dir "finetune/cym_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_mk_100M_full/best_model" \
    --target_lang mkd_Cyrl \
    --output_dir "finetune/mkd_Cyrl_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_lv_100M_full/best_model" \
    --target_lang lvs_Latn \
    --output_dir "finetune/lvs_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_lt_100M_full/best_model" \
    --target_lang lit_Latn \
    --output_dir "finetune/lit_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_sl_100M_full/best_model" \
    --target_lang slv_Latn \
    --output_dir "finetune/slv_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_sk_100M_full/best_model" \
    --target_lang slk_Latn \
    --output_dir "finetune/slk_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_ne_100M_full/best_model" \
    --target_lang npi_Deva \
    --output_dir "finetune/npi_Deva_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_et_100M_full/best_model" \
    --target_lang ekk_Latn \
    --output_dir "finetune/ekk_Latn_100M_full"


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1_ka_100M_full/best_model" \
    --target_lang kat_Geor \
    --output_dir "finetune/kat_Geor_100M_full"

