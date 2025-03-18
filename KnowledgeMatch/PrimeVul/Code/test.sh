beam_size=10
output_size=$beam_size
batch_size=1
model=Loss
input_dir=../Data
output_dir=Patch/beam_size_$beam_size
model_dir=Model

mkdir -p $output_dir

python test.py \
        --model_name_or_path $model_dir/$model \
        --test_filename $input_dir/test.jsonl,$input_dir/test-key.jsonl \
        --output_dir $output_dir \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log