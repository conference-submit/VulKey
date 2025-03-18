beam_size=100
output_size=$beam_size
batch_size=1
max_input_size=512
input_dir=primevul_data
model_dir=result
output_dir=../../Data/Train_Datasets/PrimeVul/FT-VUL/VulRepair/temperature/beam_size_$beam_size
model=Loss

mkdir -p $output_dir

python test_v4j.py \
        --model_name_or_path $model_dir/$model \
        --test_filename $input_dir/test.jsonl \
        --output_dir $output_dir \
        --max_source_length $max_input_size \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log