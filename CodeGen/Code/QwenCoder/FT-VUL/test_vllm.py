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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict, PeftModel
from vllm import LLM, SamplingParams

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

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

def check_len(src_line, tokenizer, max_length, i, output_len):
    input_tokens = tokenizer.tokenize(src_line)
    # print(f"{i}:{len(input_tokens)}")
    if len(input_tokens) < max_length-output_len:
        return src_line
    else:
        # print('--------------------------------------')
        while True:
            input_tokens = tokenizer.tokenize(src_line)

            if len(input_tokens) > max_length - output_len:
                try:

                    bug_start_loc = input_tokens.index('//')
                    bug_end_loc = len(input_tokens) - 1 - input_tokens[::-1].index('//')
                except ValueError:
                    raise ValueError("Input text must contain both '// bug_start' and '// bug_end'")

                bug_section_length = bug_end_loc - bug_start_loc

                if bug_section_length > max_length - output_len:
                    # print(src_line)
                    break

                remaining_length = max_length - output_len - bug_section_length
                half_remaining_length = remaining_length // 2

                if bug_start_loc + half_remaining_length < len(input_tokens):
                    src_line = src_line[:len(src_line) - 2]
                elif bug_start_loc - half_remaining_length > 0:
                    src_line = src_line[1:]
                else:
                    src_line = src_line[1:]
            else:
                break

        # logger.info('%s, %s', str(i), str(len(input_tokens)))
        # print(f"{i}:{len(input_tokens)}")

        return src_line



def model_inference():
    logger.info("-----------------------------------")
    logger.info("    Load Tokenizer and Model...    ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True)
    

    model = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        max_model_len=args.max_source_length,
        dtype=torch.bfloat16,
        gpu_memory_utilization = 0.9
    )

    # print_trainable_parameters(model)
    
    logger.info("-----------------------------------")
    logger.info("            Start Test...          ")
    
    test_filename = args.test_filename
    

    with open(test_filename, 'r') as data_file:
        data_lines = data_file.readlines()
    data_size = len(data_lines)
    
    pre_list = []
    tgt_list = []
    good = 0
    i = 0
    


    with open(test_filename, 'r') as f:
        logger.info("Data Size: %s", str(data_size)) 
        
        for line in f:
            i += 1

            # if i <= 471:
            #     continue
            
            test_dic = json.loads(line)         

            src_line = test_dic['source']
            
            tgt_line = test_dic['target']
            
            tgt_list.append(str([tgt_line])+'\n')

            output_len = max(len(tokenizer.tokenize(tgt_line + '\n[bug_function]\n' + '\n[fix_code]\n' )) + 10, 128)
            src_line = check_len(src_line, tokenizer, args.max_source_length, i, output_len)

            input_text = '<|im_start|>user\n' + '\n[bug_function]\n' + src_line + '<|im_end|>\n' + '<|im_start|>\n' + '\n[fix_code]\n'
     
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
                    tem_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                    for tem_size in tem_list:
                        sampling_params = SamplingParams(
                            max_tokens=args.max_target_length,
                            n=args.output_size,
                            temperature=tem_size,
                            top_p=1.0,
                            stop_token_ids=[eos_id],
                        )
                        outputs = model.generate(input_text, sampling_params=sampling_params)
                        """
                        generated_ids = model.generate(
                            input_ids=input_ids, 
                            max_length=args.max_source_length, 
                            # num_beams=args.beam_size, 
                            num_return_sequences=args.output_size, 
                            # early_stopping=True,
                            temperature=tem_size,
                            top_p=1.0,
                            do_sample=True,
                            pad_token_id=eos_id, 
                            eos_token_id=eos_id)
                        """ 

                        for request_output in outputs:
                            prompt = request_output.prompt 
                            generated_texts = [output.text for output in request_output.outputs] 
                            #print(f"Prompt: {prompt!r}")
                            for idx, text in enumerate(generated_texts):
                                #print(f"Generated text {idx + 1}: {text!r}")  
                                pre_out.append(text)    
                except Exception as e:
                    pre_out = []
                    pre_out.append('// Error\n')
                    logger.info('%s', str(e))
                
                EM_result = 'bad'
                for pre_one in pre_out:
                    pre_txt = pre_one.replace('// fix_start\n','').replace('// fix_end\n', '').replace('\n','')
                    pre_txt = re.sub(r'\s+', '', pre_txt)

                    output_txt = output_text.replace('// fix_start\n','').replace('// fix_end\n', '').replace('\n','')
                    output_txt = re.sub(r'\s+', '', output_txt)

                    if pre_txt.strip() == output_txt.strip():
                        good += 1
                        EM_result = 'good'
                        break
                    
                # logger.info(" %d/%d, Pre_Output = %s", i, data_size, str(pre_out))    
                logger.info(" %d/%d, Pre_EM = %s, Input_len = %d, Beam_size = %s, Sample_size = %s, Pre_Fix = %s", i, data_size, str(EM_result), input_ids.size(1), str(args.beam_size), str(len(pre_out)), str(pre_out)) 
                pre_list.append(str(pre_out) + '\n')

    # logger.info(" Good = %d , Repair Accuracy = %s", good, str(good/data_size))           
            if i % 5 == 0:
                with open(args.output_dir+'/pre_file.txt', 'w') as pre_file, open(args.output_dir+'/tgt_file.txt', 'w') as tgt_file:
                    for pre, tgt in zip(pre_list, tgt_list):
                        pre_file.write(pre)
                        tgt_file.write(tgt)

    
    logger.info(" Good = %d", good)
    logger.info(" All = %d", i)
    logger.info(" Repair Accuracy = %s", str(good/i))   



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
    

    
