import codecs
import os
import re
import sys
import json
import torch
import logging
import argparse
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5Model
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from transformers import AutoTokenizer, T5ForConditionalGeneration


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    
    parser.add_argument("--model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--test_filename", type=str, default="TRANSFER")
    parser.add_argument("--output_dir", type=str, default="TRANSFER")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    
    return parser.parse_args()

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

def check_len(src_line, tokenizer, max_length, i, output_len = 32):
    input_tokens = tokenizer.tokenize(src_line)
    #print(f"{i}:{len(input_tokens)}")
    
    if len(input_tokens) < max_length - output_len:
        return src_line
    
    # Find the first occurrence of '//' and the last occurrence of '//' in the tokenized input
    try:
        bug_start_loc = input_tokens.index('//')
        bug_end_loc = len(input_tokens) - 1 - input_tokens[::-1].index('//')
    except ValueError:
        raise ValueError("Input text must contain both '//' markers")

    bug_section_length = bug_end_loc - bug_start_loc

    if bug_section_length > max_length - output_len:
        # print(f"{i}:{bug_section_length}")
        return src_line

    remaining_length = max_length - output_len - bug_section_length
    half_remaining_length = remaining_length // 2

    # Calculate the new start and end positions for truncation
    new_start = max(0, bug_start_loc - half_remaining_length)
    new_end = min(len(input_tokens), bug_end_loc + half_remaining_length)

    # Truncate the source line based on the new start and end positions
    truncated_tokens = input_tokens[new_start:new_end]
    src_line = tokenizer.convert_tokens_to_string(truncated_tokens)

    # Re-tokenize the truncated source line to check the length
    input_tokens = tokenizer.tokenize(src_line)
    #print(f"{i}:{len(input_tokens)}")

    return src_line


def model_inference():
    logger.info("-----------------------------------")
    logger.info("    Load Tokenizer and Model...    ")



    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, device_map='auto')

    # tokenizer.add_tokens(['InsertMemset', 'InsertReleaseResource', 'InsertCastStatement', 'InsertCastChecker', 'InsertRangeChecker', 'InsertNullPointerChecker', 'InsertMissedStatement', 'MutateControlStatement', 'InsertConditionalExpression', 'MutateConditionalExpression', 'MutateDataType', 'MutateLiteralExpression', 'InsertMethodInvocationExpression', 'MutateMethodInvocationExpression', 'MutateReturnStatement', 'MutateVariable', 'MoveStatement', 'RemoveBuggyStatement', 'Insert Variable', 'OtherTemplate'])
    # tokenizer.add_tokens(['CWE-16', 'CWE-17', 'CWE-19', 'CWE-20', 'CWE-22', 'CWE-59', 'CWE-61', 'CWE-74', 'CWE-77', 'CWE-78', 'CWE-79', 'CWE-89', 'CWE-93', 'CWE-94', 'CWE-116', 'CWE-119', 'CWE-120', 'CWE-121', 'CWE-122', 'CWE-125', 'CWE-129', 'CWE-131', 'CWE-134', 'CWE-172', 'CWE-189', 'CWE-190', 'CWE-191', 'CWE-193', 'CWE-200', 'CWE-203', 'CWE-209', 'CWE-212', 'CWE-241', 'CWE-252', 'CWE-254', 'CWE-255', 'CWE-264', 'CWE-269', 'CWE-273', 'CWE-275', 'CWE-276', 'CWE-281', 'CWE-284', 'CWE-285', 'CWE-287', 'CWE-288', 'CWE-290', 'CWE-294', 'CWE-295', 'CWE-307', 'CWE-310', 'CWE-311', 'CWE-320', 'CWE-326', 'CWE-327', 'CWE-331', 'CWE-345', 'CWE-346', 'CWE-347', 'CWE-352', 'CWE-354', 'CWE-361', 'CWE-362', 'CWE-369', 'CWE-388', 'CWE-399', 'CWE-400', 'CWE-401', 'CWE-404', 'CWE-415', 'CWE-416', 'CWE-417', 'CWE-434', 'CWE-444', 'CWE-459', 'CWE-476', 'CWE-502', 'CWE-522', 'CWE-532', 'CWE-552', 'CWE-565', 'CWE-601', 'CWE-611', 'CWE-613', 'CWE-617', 'CWE-639', 'CWE-662', 'CWE-664', 'CWE-665', 'CWE-667', 'CWE-668', 'CWE-670', 'CWE-672', 'CWE-674', 'CWE-681', 'CWE-682', 'CWE-697', 'CWE-703', 'CWE-704', 'CWE-706', 'CWE-707', 'CWE-732', 'CWE-754', 'CWE-755', 'CWE-763', 'CWE-770', 'CWE-772', 'CWE-787', 'CWE-798', 'CWE-824', 'CWE-834', 'CWE-835', 'CWE-843', 'CWE-862', 'CWE-863', 'CWE-908', 'CWE-909', 'CWE-918', 'CWE-924', 'CWE-1021'])
    # print_trainable_parameters(model)
    
    logger.info("-----------------------------------")
    logger.info("            Start Test...          ")
    
    assert len(args.test_filename.split(','))==2
    src_filename = args.test_filename.split(',')[0]
    trg_filename = args.test_filename.split(',')[1]
    
    
    with open(src_filename, 'r') as data_file:
        data_lines = data_file.readlines()
    data_size = len(data_lines)
    
    pre_list = []
    tgt_list = []
    good = 0
    i = 0


    with open(src_filename, 'r') as f1,open(trg_filename, 'r') as f2:
        # all_data = f2.readlines()
        # data_size = len(all_data)
        # print('Dataset Size:')
        # print(len(f1), len(f2))
        logger.info("Data Size: %s", str(data_size)) 
        for line1 in f1:
            i += 1
            # if i <= 558:
            #     continue

            src_dic = json.loads(line1)
            # tgt_dic = json.loads(line2)
            

            src_line = src_dic['source']
            src_line = check_len(src_line, tokenizer, args.max_source_length, i, 32)
            src_line = src_dic['cwe']+ ' ' + src_line
            tgt_line = ' '

            
            tgt_list.append(str([tgt_line])+'\n')
            
            
            input_text = src_line
     
            # input_text = src_line.replace('<BUGS>','<commit_before>').replace('<BUGE>','<commit_end>') + '\n' + '// repair template: \n'
            output_text = tgt_line
            
            # logger.info("Input Text: %s", str(input_text))   

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)
            pre_out = []

            if input_ids.size(1) >= args.max_source_length:
                pre_out.append('Input Too Long')
                logger.info(" %d/%d , Input_Len = %d, Pre_Output = %s", i, data_size, input_ids.size(1), str(['Input Too Long']))
                pre_list.append(str(pre_out) + '\n')
            else:

                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                try:
                    generated_ids = model.generate(
                        input_ids=input_ids, 
                        max_new_tokens=args.max_target_length, 
                        num_beams=args.beam_size, 
                        num_return_sequences=args.output_size, 
                        early_stopping=True,
                        pad_token_id=eos_id, eos_token_id=eos_id)
                               
                    for generated_id in generated_ids:
                        generated_text = tokenizer.decode(generated_id, skip_special_tokens=False)
                        text = generated_text
                        text = text.replace('</s>','').replace('<pad>','').replace('<s>','')
                        pre_out.append(text)
                except Exception as e:
                    pre_out = []
                    pre_out.append('// Error\n')
                    logger.info('%s', str(e))
                
                EM_result = 'bad'
                for pre_one in pre_out:
                    pre_txt = pre_one.replace('\n','')
                    pre_txt = re.sub(r'\s+', '', pre_txt)

                    output_txt = output_text.replace('\n','')
                    output_txt = re.sub(r'\s+', '', output_txt)

                    if pre_txt.strip() == output_txt.strip():
                        good += 1
                        EM_result = 'good'
                        break
                    
                # logger.info(" %d/%d, Pre_Output = %s", i, data_size, str(pre_out))    
                logger.info(" %d/%d, Pre_EM = %s, Input_Len = %s, Pre_Fix = %s", i, data_size, str(EM_result), str(input_ids.size(1)), str(pre_out)) 
                pre_list.append(str(pre_out) + '\n')
    
    logger.info(" Good = %d", good)
    logger.info(" All = %d", i)
    logger.info(" Repair Accuracy = %s", str(good/i))
    # logger.info(" Good = %d , Repair Accuracy = %s", good, str(good/data_size))           
    with open(args.output_dir+'/pre_file.txt', 'w') as pre_file, open(args.output_dir+'/tgt_file.txt', 'w') as tgt_file:
        
        for pre, tgt in zip(pre_list, tgt_list):
            pre_file.write(pre)
            tgt_file.write(tgt)
    



if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # args.device = device
    # logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    model_dir = args.model_name_or_path
    input_file = args.test_filename
    output_dir = args.output_dir
        
    model_inference()
    

    
