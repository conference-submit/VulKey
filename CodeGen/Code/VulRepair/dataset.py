import torch
import codecs
import random
import json
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path, tokenizer, max_length, max_output_length, shuffle=False, load_range=None):
        self.data = []
        self.max_length = max_length
        

        train_filename = file_path

        num = 0
        with open(train_filename, 'r') as f:
            # print('Dataset Size:')
            # print(len(f1), len(f2))
            for line in f:
                test_dic = json.loads(line)   
            
                src_line = test_dic['source']
                tgt_line = test_dic['target']
            
                num += 1
                
                inputs = src_line 
                outputs = tgt_line + tokenizer.eos_token

                inputs_ids = tokenizer.encode(inputs, truncation=True, max_length=max_length,  return_tensors='pt')
                outputs_ids = tokenizer.encode(outputs, truncation=True, max_length=max_output_length,  return_tensors='pt')

                if inputs_ids.size(1) > max_length or outputs_ids.size(1) > max_length:
                    print(f'Input Too Long: {inputs.size(1)}, Index: {num}')
                    continue

                self.data.append({
                    'input_ids': inputs_ids,
                    'labels': outputs_ids,
                    'attention_mask': torch.ones(inputs_ids.size()).long()
                })

        
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
