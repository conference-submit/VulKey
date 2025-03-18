import torch
import codecs
import random
import json
from transformers import AutoTokenizer

def check_len(src_line, tokenizer, max_length, i, output_len):
    input_tokens = tokenizer.tokenize(src_line)
    # print(f"{i}:{len(input_tokens)}")
    
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
        #print(f"{i}:{bug_section_length}")
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
    # print(f"{i}:{len(input_tokens)}")

    return src_line



class Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path, tokenizer, max_length, shuffle=False, load_range=None):
        self.data = []
        self.max_length = max_length
        
        assert len(file_path.split(','))==2
        src_filename = file_path.split(',')[0]
        trg_filename = file_path.split(',')[1]

        num = 0
        with open(src_filename, 'r') as f1,open(trg_filename, 'r') as f2:
            # print('Dataset Size:')
            # print(len(f1), len(f2))
            for line1,line2 in zip(f1,f2):
                src_dic = json.loads(line1)
                tgt_dic = json.loads(line2)
            
                num += 1
                src_line = src_dic['source']
                src_line = check_len(src_line, tokenizer, max_length, num, 32)
                src_line = src_dic['cwe']+ ' ' + src_line 

                template = list(tgt_dic['template'].keys())[0].replace(" ", "") + '\nKeywords:\n' + list(tgt_dic['template'].values())[0]

                
                # inputs = '<commit_before>\n' + src_line + '\n<commit_after>\n' + tgt_line + tokenizer.eos_token
                # outputs = tgt_line + tokenizer.eos_token

                # if num % 1000 == 0:
                #     print(num)
                
                # if len(tokenizer.tokenize(src_line)) > 768 :
                #     continue
                

                # try:
                #     src_line = check_len(src_line, tokenizer, max_length)
                # except:
                #     continue

                
                inputs = src_line 
                outputs = template.strip() + tokenizer.eos_token


                inputs = tokenizer.encode(inputs, return_tensors='pt')
                outputs = tokenizer.encode(outputs, return_tensors='pt')

                if inputs.size(1) > max_length or outputs.size(1) > max_length:
                    print(f'Input Too Long: {inputs.size(1)}, Index: {num}')
                    continue

                self.data.append({
                    'input_ids': inputs,
                    'labels': outputs,
                    'attention_mask': torch.ones(inputs.size()).long()
                })

                # if len(self.data) % 10000 == 0:
                #     print('finish loading:', len(self.data))
                
                # if load_range is not None and len(self.data) == load_range[1]:
                #     break
        
        if shuffle:
            random.seed(7)
            random.shuffle(self.data)

        print(file_path, 'total size:', len(self.data))
        if load_range is not None:
            self.data = self.data[load_range[0]: ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]


# def custom_collate(batch):
#     batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
#     max_input_len = max([b['input_ids'].size(1) for b in batch])
#     max_output_len = max([b['labels'].size(1) for b in batch])
#     for b in batch:
#         batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).fill_(2).long()], dim=1))
#         batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
#         batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
#     batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
#     batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
#     batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
#     return batch_data

def custom_collate(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_input_len = max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data
