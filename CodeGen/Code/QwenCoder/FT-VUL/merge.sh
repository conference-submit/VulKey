max_input_size=2048
output_dir=result
# mkdir -p $output_dir

python merge.py \
        --base_model_name_or_path ../FT-BUG/result/-merged \
        --peft_model_path $output_dir/  \
        --push_to_hub \