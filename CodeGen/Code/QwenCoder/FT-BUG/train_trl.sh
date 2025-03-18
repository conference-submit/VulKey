#lr=5e-5
lr=3e-4
beam_size=1
epoch=1
batch_size=4
input_dir=../../../Data/Train_Datasets/Transfer
output_dir=result
model_dir=Qwen/Qwen2.5-Coder-32B

mkdir -p $output_dir

python train_trl.py \
        --model_name_or_path $model_dir \
        --train_filename $input_dir/src-train.jsonl,$input_dir/tgt-train.jsonl \
        --dev_filename $input_dir/src-val.jsonl,$input_dir/tgt-val.jsonl \
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --eval_step 1 \
        2>&1 | tee $output_dir/train.log