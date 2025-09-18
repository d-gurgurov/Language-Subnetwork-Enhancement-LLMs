#!/bin/bash

pip install vllm 
pip install -U accelerate huggingface_hub datasets
pip install numpy==1.26.4

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

# ========= SINGLE EVAL. ==========

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_mt_100M/best_model" \
    --target_lang mlt_Latn \
    --output_dir "finetune_nemo/mlt_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_mt_100M_random/best_model" \
    --target_lang mlt_Latn \
    --output_dir "finetune_nemo/mlt_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_af_100M/best_model" \
    --target_lang afr_Latn \
    --output_dir "finetune_nemo/afr_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_af_100M_random/best_model" \
    --target_lang afr_Latn \
    --output_dir "finetune_nemo/afr_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_is_100M/best_model" \
    --target_lang isl_Latn \
    --output_dir "finetune_nemo/isl_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_is_100M_random/best_model" \
    --target_lang isl_Latn \
    --output_dir "finetune_nemo/isl_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_cy_100M/best_model" \
    --target_lang cym_Latn \
    --output_dir "finetune_nemo/cym_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_cy_100M_random/best_model" \
    --target_lang cym_Latn \
    --output_dir "finetune_nemo/cym_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_mk_100M/best_model" \
    --target_lang mkd_Cyrl \
    --output_dir "finetune_nemo/mkd_Cyrl_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_mk_100M_random/best_model" \
    --target_lang mkd_Cyrl \
    --output_dir "finetune_nemo/mkd_Cyrl_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_lv_100M/best_model" \
    --target_lang lvs_Latn \
    --output_dir "finetune_nemo/lvs_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_lv_100M_random/best_model" \
    --target_lang lvs_Latn \
    --output_dir "finetune_nemo/lvs_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_lt_100M/best_model" \
    --target_lang lit_Latn \
    --output_dir "finetune_nemo/lit_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_lt_100M_random/best_model" \
    --target_lang lit_Latn \
    --output_dir "finetune_nemo/lit_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_sl_100M/best_model" \
    --target_lang slv_Latn \
    --output_dir "finetune_nemo/slv_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_sl_100M_random/best_model" \
    --target_lang slv_Latn \
    --output_dir "finetune_nemo/slv_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_sk_100M/best_model" \
    --target_lang slk_Latn \
    --output_dir "finetune_nemo/slk_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_sk_100M_random/best_model" \
    --target_lang slk_Latn \
    --output_dir "finetune_nemo/slk_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_et_100M/best_model" \
    --target_lang ekk_Latn \
    --output_dir "finetune_nemo/ekk_Latn_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_et_100M_random/best_model" \
    --target_lang ekk_Latn \
    --output_dir "finetune_nemo/ekk_Latn_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_ka_100M/best_model" \
    --target_lang kat_Geor \
    --output_dir "finetune_nemo/kat_Geor_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_ka_100M_random/best_model" \
    --target_lang kat_Geor \
    --output_dir "finetune_nemo/kat_Geor_100M_random"


python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_ne_100M/best_model" \
    --target_lang npi_Deva \
    --output_dir "finetune_nemo/npi_Deva_100M"

python eval_finetune.py \
    --base_model "mistralai/Mistral-Nemo-Base-2407" \
    --finetuned_model "finetune_nemo/Mistral-Nemo-Base-2407_5_ne_100M_random/best_model" \
    --target_lang npi_Deva \
    --output_dir "finetune_nemo/npi_Deva_100M_random"

