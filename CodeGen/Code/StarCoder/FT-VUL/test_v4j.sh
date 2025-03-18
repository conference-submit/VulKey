beam_size=100
output_size=$beam_size
batch_size=1
model=Epoch_1
input_dir=../../../Data/Test_Benchmarks/Vul4J
model_dir=result_StarCoder-15B/
output_dir=../../../Data/Test_Benchmarks/Vul4J/FT-VUL/StarCoder/temperature/beam_size_$beam_size
mkdir -p $output_dir

python test_v4j.py \
        --model_name_or_path $model_dir/$model/-merged \
        --test_filename $input_dir/src-test.jsonl,$input_dir/tgt-test.jsonl \
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log