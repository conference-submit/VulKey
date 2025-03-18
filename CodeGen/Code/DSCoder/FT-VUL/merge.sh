max_input_size=2048
output_dir=result_DSCoder-16B/
model_dir=../../../Code/DeepSeek/FT-BUG/result_DSCoder-16B/-merged


# mkdir -p $output_dir

python merge.py \
        --base_model_name_or_path $model_dir \
        --peft_model_path $output_dir  \
        --push_to_hub \