lr=5e-5
beam_size=1
epoch=1
batch_size=1
input_dir=../../..NTR/Data/Train_Datasets/PrimeVul
output_dir=result_CodeLlama-70B/
model_dir=LLM4APR/CodeLlama-70B_for_NMT


mkdir -p $output_dir

python train.py \
        --model_name_or_path $model_dir/Epoch_1/-merged \
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