import os
import time
import logging
import argparse
import traceback
import torch
import torch.nn as nn
from dataset import Dataset as CustomDataset, custom_collate
from datasets import Dataset as HFDataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq
from transformers import get_cosine_schedule_with_warmup, Adafactor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict, PeftModel
from accelerate import infer_auto_device_map
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import pdb

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--dataset_name", type=str, default="Transfer")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=10000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    
    parser.add_argument("--train_filename", type=str, default="/src")
    parser.add_argument("--dev_filename", type=str, default="/tgt")
    parser.add_argument("--output_dir", type=str, default="/save_model")
    
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=1000)
    
    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str, default="completion")
    
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=5, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=200, type=int)

    return parser.parse_args()


def fine_tune():
    logger.info("---------------------------")
    print('Load Tokenizer and Model...')
    
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', use_auth_token=True)

    tokenizer.add_tokens(['[bug_function]', '[fix_code]'])
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/DeepSeek-Coder-V2-Lite-Base',
        #load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args.model_name_or_path, device_map='auto', is_trainable = True)
    
    for param in model.parameters():
        param.requires_grad = True

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'
               ]   
    )

    # model = get_peft_model(model, lora_config)

    # print_trainable_parameters(model)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_source_length,
        dataloader_num_workers=32,
        learning_rate=args.learning_rate,
        logging_steps=args.log_freq,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        save_strategy="steps",
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=True,
        lr_scheduler_type=args.lr_scheduler_type,
        # warmup_steps=args.num_warmup_steps,
        # weight_decay=args.weight_decay,
        report_to="wandb",
        run_name="DSCoder-F",
        save_safetensors=True,
        group_by_length=True,
        optim="adafactor",
        resume_from_checkpoint=True,
        use_liger=True
    )

    train_dataset = CustomDataset(
        args.train_filename, 
        tokenizer,
        max_length=args.max_source_length,
        shuffle=True
    )
    
    eval_dataset = CustomDataset(
        args.dev_filename,
        tokenizer,
        max_length=args.max_source_length
    )

    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # 初始化Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate, 
        formatting_func=None,
        peft_config=lora_config,
        tokenizer=tokenizer
    )

    print(tokenizer.decode(trainer.train_dataset[5]["input_ids"][0]))
    labels = trainer.train_dataset[5]["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.squeeze().tolist() 

    decodable_labels = [tokenizer.pad_token_id if x == -100 else x for x in labels]
    print(tokenizer.decode(decodable_labels))

    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # args.device = device
    # logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    

    device_ids = [5,6]

    fine_tune()
