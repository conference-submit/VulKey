lr=4e-5
beam_size=1
epoch=75
batch_size=16
input_dir=primevul_data
output_dir=result


mkdir -p $output_dir

python train.py \
        --model_name_or_path Salesforce/codet5-base \
        --train_filename $input_dir/train.jsonl \
        --dev_filename $input_dir/valid.jsonl\
        --output_dir $output_dir \
        --max_source_length 512 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --eval_step 1 \
        2>&1 | tee $output_dir/train.log