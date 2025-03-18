#lr=5e-5
lr=5e-5
beam_size=1
epoch=1
batch_size=2
input_dir=../../../Data/Train_Datasets/PrimeVul
output_dir=result_DSCoder-16B
model_dir=../../../Code/DeepSeek/FT-BUG/result_DSCoder-16B


mkdir -p $output_dir

python train_trl.py \
        --model_name_or_path $model_dir \
        --train_filename $input_dir/train.jsonl \
        --dev_filename $input_dir/val.jsonl\
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --eval_step 1 \
        2>&1 | tee $output_dir/train.log