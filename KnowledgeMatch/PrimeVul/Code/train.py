import os
import time
import logging
import argparse
import traceback
import torch
import torch.nn as nn
from dataset import Dataset, custom_collate
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup, Adafactor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from accelerate import infer_auto_device_map
from transformers import AutoTokenizer, T5ForConditionalGeneration


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,4,6'

# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )

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
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
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
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()


def validation_step(epoch, model, validation_loader, save_dir, best_loss, parallel=False):
    # print('-------start validation--------')
    logger.info("-------start validation--------")
    all_data = len(validation_loader)
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
            loss = output.loss
            if i%200 == 0:
                logger.info(" %d/%d , Val_Loss = %s", i, all_data, loss.mean().item())
            validation_loss.append(loss.mean().item())
    
    test_loss = sum(validation_loss) / len(validation_loss)
    # print('validation loss:', round(sum(validation_loss) / len(validation_loss), 4))
    logger.info("Validation Loss = %s", test_loss)
    
    # if not parallel:
    #     model.module.save_pretrained(save_dir+'/Epoch_'+str(epoch+1))
    # else:
    #     model.save_pretrained(save_dir+'/Epoch_'+str(epoch+1))
    
    
    
    if test_loss < best_loss:
        best_loss = test_loss
        logger.info("Current is Best Loss = %s", test_loss)
        if not parallel:
            model.module.save_pretrained(save_dir+'/Loss')
        else:
            model.save_pretrained(save_dir+'/Loss')
    else:
        logger.info("Current Not Best Loss, Current Loss = %s, Best Loss is = %s", test_loss, best_loss)
     
    if epoch == epochs-1:
        # print('This is Last Checkpoint, Currect Loss=' + str(test_loss))
        logger.info("This is Last Checkpoint, Current Loss = %s, Best Loss is = %s", test_loss, best_loss)
        if not parallel:
            model.module.save_pretrained(save_dir+'/Last')
        else:
            model.save_pretrained(save_dir+'/Last')
    
    model.train()
    logger.info("---------end validation--------")
    
    return best_loss


def fine_tune(training_file, validation_file, epochs, batch_size, save_dir, parallel=True, load_range=None):
    logger.info("---------------------------")
    print('Load Tokenizer and Model...')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, device_map='auto')

    tokenizer.add_tokens(['<keyword>'])
    #tokenizer.add_tokens(['InsertMemset', 'InsertReleaseResource', 'InsertCastStatement', 'InsertCastChecker', 'InsertRangeChecker', 'InsertNullPointerChecker', 'InsertMissedStatement', 'MutateControlStatement', 'InsertConditionalExpression', 'MutateConditionalExpression', 'MutateDataType', 'MutateLiteralExpression', 'InsertMethodInvocationExpression', 'MutateMethodInvocationExpression', 'MutateReturnStatement', 'MutateVariable', 'MoveStatement', 'RemoveBuggyStatement', 'InsertVariable', 'OtherTemplate'])
    #tokenizer.add_tokens(['CWE-16', 'CWE-17', 'CWE-19', 'CWE-20', 'CWE-22', 'CWE-59', 'CWE-61', 'CWE-74', 'CWE-77', 'CWE-78', 'CWE-79', 'CWE-89', 'CWE-93', 'CWE-94', 'CWE-116', 'CWE-119', 'CWE-120', 'CWE-121', 'CWE-122', 'CWE-125', 'CWE-129', 'CWE-131', 'CWE-134', 'CWE-172', 'CWE-189', 'CWE-190', 'CWE-191', 'CWE-193', 'CWE-200', 'CWE-203', 'CWE-209', 'CWE-212', 'CWE-241', 'CWE-252', 'CWE-254', 'CWE-255', 'CWE-264', 'CWE-269', 'CWE-273', 'CWE-275', 'CWE-276', 'CWE-281', 'CWE-284', 'CWE-285', 'CWE-287', 'CWE-288', 'CWE-290', 'CWE-294', 'CWE-295', 'CWE-307', 'CWE-310', 'CWE-311', 'CWE-320', 'CWE-326', 'CWE-327', 'CWE-331', 'CWE-345', 'CWE-346', 'CWE-347', 'CWE-352', 'CWE-354', 'CWE-361', 'CWE-362', 'CWE-369', 'CWE-388', 'CWE-399', 'CWE-400', 'CWE-401', 'CWE-404', 'CWE-415', 'CWE-416', 'CWE-417', 'CWE-434', 'CWE-444', 'CWE-459', 'CWE-476', 'CWE-502', 'CWE-522', 'CWE-532', 'CWE-552', 'CWE-565', 'CWE-601', 'CWE-611', 'CWE-613', 'CWE-617', 'CWE-639', 'CWE-662', 'CWE-664', 'CWE-665', 'CWE-667', 'CWE-668', 'CWE-670', 'CWE-672', 'CWE-674', 'CWE-681', 'CWE-682', 'CWE-697', 'CWE-703', 'CWE-704', 'CWE-706', 'CWE-707', 'CWE-732', 'CWE-754', 'CWE-755', 'CWE-763', 'CWE-770', 'CWE-772', 'CWE-787', 'CWE-798', 'CWE-824', 'CWE-834', 'CWE-835', 'CWE-843', 'CWE-862', 'CWE-863', 'CWE-908', 'CWE-909', 'CWE-918', 'CWE-924', 'CWE-1021'])
    model.resize_token_embeddings(len(tokenizer))
   
    logger.info("---------------------------")
    logger.info("Load Dataset...")
    
    training_dataset = Dataset(training_file, tokenizer, max_length=args.max_source_length, shuffle=True, load_range=load_range)
    validation_dataset = Dataset(validation_file, tokenizer, max_length=args.max_source_length, load_range=None)
    logger.info("Load Dataset Success.")

    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-4, momentum=0.9)
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(epochs * len(training_loader))
    )

    accelerator = Accelerator()
    device = accelerator.device

    model, optimizer, training_loader, validation_loader = accelerator.prepare(
        model, optimizer, training_loader, validation_loader
    )
    
    # print('Model train...')
    logger.info("Model train...")
    
    #Start training
    logger.info("***** Running training *****")
    logger.info("      Batch size = %d", args.train_batch_size)
    logger.info("      Num epoch = %d", args.num_train_epochs)
    logger.info("****************************")
    
    best_loss = 100000000000
    
    for epoch in range(epochs):
        logger.info("  epoch = %d", epoch)
        model.train()
        training_loss = []
        start_time = time.time()
        oom = 0
        all_data = len(training_loader)

        for i, data in enumerate(training_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            try:
                optimizer.zero_grad()
                
                # output = model(input_ids=data['input_ids'], decoder_input_ids=data['labels'], return_dict=True)
                output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                
                loss = output.loss
                
                loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                optimizer.step()
                scheduler.step()

               

                training_loss.append(loss.mean().item())
                if i % 200 == 0:
                    logger.info("Epoch: %d, Train_Step: %d/%d, Train_Loss = %s", epoch, i, all_data, round(sum(training_loss) / len(training_loss), 4))
            except Exception as e:
                # print(str(e))
                logger.info(str(e))
                
                if 'out of memory' in str(e):
                    oom += 1
                model.zero_grad()
                optimizer.zero_grad()
                scheduler.step()
                del data

                torch.cuda.empty_cache()

        if epoch % args.eval_step == 0:
            logger.info('epoch: {}, step: {}/{}, loss: {}, lr: {}, oom: {}, time: {}s'.format(
                epoch, i, len(training_loader),
                round(sum(training_loss) / len(training_loss), 4),
                round(scheduler.get_last_lr()[0], 7), oom,
                int(time.time() - start_time)
            ))
            start_time = time.time()
            oom = 0
        
            best_loss=validation_step(epoch, model, validation_loader, save_dir, best_loss, parallel=parallel)
        
        # if epoch % args.eval_step == 0:
        #     best_loss=validation_step(model, validation_loader, save_dir, best_loss=-1000000, parallel=parallel)


if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # args.device = device
    # logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    os.makedirs(args.output_dir, exist_ok=True)

    device_ids = [1]

    training_file = args.train_filename
    validation_file = args.dev_filename
    vocabulary_file = args.model_name_or_path
    pretrained_file = args.model_name_or_path
    
    epochs = args.num_train_epochs
    batch_size = args.train_batch_size
    save_dir = args.output_dir

    fine_tune(
        training_file, validation_file, epochs, batch_size, save_dir, parallel=True, load_range=None
    )
