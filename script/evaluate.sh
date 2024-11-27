!CUDA_VISIBLE_DEVICES=0 python /content/LoRAPrune/inference.py \
    --base_model "yahma/llama-7b-hf" \
    --dataset /content/drive/MyDrive/Masterproef/notebooks/data/hf_dataset_eval \
    --lora_weights /content/drive/MyDrive/Masterproef/model \
    --cutoff_len 4096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"