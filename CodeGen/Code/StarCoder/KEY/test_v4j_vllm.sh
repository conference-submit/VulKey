beam_size=10
output_size=$beam_size
batch_size=1
# top_n=10
max_input_size=2048
input_dir=../../../Data/Test_Benchmarks/Vul4J
model_dir=../FT-VUL/result_StarCoder-15B
output_dir=../../../Data/Test_Benchmarks/Vul4J/KEY/StarCoder/temperature/beam_size_$beam_size

mkdir -p $output_dir

python test_v4j_vllm.py \
        --model_name_or_path $model_dir/Epoch_1/-merged \
        --test_filename $input_dir/test.jsonl,$input_dir/test-tokens.jsonl \
        --output_dir $output_dir \
        --max_source_length $max_input_size \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log