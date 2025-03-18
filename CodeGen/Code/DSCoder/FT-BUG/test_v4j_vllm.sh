beam_size=100
output_size=$beam_size
batch_size=1
# top_n=10
max_input_size=2048
input_dir=/apdcephfs/private_sakurapeng/linsayli/NTR/Data/Test_Benchmarks/Vul4J
# input_dir=/data3/HuangKai/Dataset/Recoder_dataset/2-Program_Repair/Recoder_test/$ts_model/top_$top_n
model_dir=result_DSCoder-16B-2
output_dir=/apdcephfs/private_sakurapeng/linsayli/NTR/Data/Test_Benchmarks/Vul4J/FT-BUG/DeepSeek/temperature/beam_size_$beam_size

mkdir -p $output_dir

python test_v4j_vllm.py \
        --model_name_or_path $model_dir/-merged \
        --test_filename $input_dir/src-test.jsonl,$input_dir/tgt-test.jsonl \
        --output_dir $output_dir \
        --max_source_length $max_input_size \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log