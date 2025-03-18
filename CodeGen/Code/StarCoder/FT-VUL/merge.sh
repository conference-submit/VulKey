output_dir=result_StarCoder-15B/

# mkdir -p $output_dir

python merge.py \
        --base_model_name_or_path LLM4APR/StarCoder-15B_for_NMT  \
        --peft_model_path $output_dir/Epoch_1/  \
        --push_to_hub \