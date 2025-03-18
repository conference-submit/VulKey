beam_size=1
output_size=$beam_size
batch_size=1
# top_n=10
max_input_size=2048
input_dir=../../../Data/Train_Datasets/PrimeVul
# input_dir=/data3/HuangKai/Dataset/Recoder_dataset/2-Program_Repair/Recoder_test/$ts_model/top_$top_n
model_dir=result/-merged
output_dir=../../../Data/Train_Datasets/PrimeVul/FT-VUL/Qwen/temperature/beam_size_$beam_size

mkdir -p $output_dir

python test_vllm.py \
        --model_name_or_path $model_dir \
        --test_filename $input_dir/test.jsonl \
        --output_dir $output_dir \
        --max_source_length $max_input_size \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log