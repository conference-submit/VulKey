max_input_size=2048
output_dir=result_CodeLlama-70B

# mkdir -p $output_dir

python merge.py \
        --base_model_name_or_path LLM4APR/CodeLlama-70B_for_NMT \
        --peft_model_path $output_dir/Epoch_1/  \
        --push_to_hub \