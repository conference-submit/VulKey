from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    return parser.parse_args()

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    tokenizer.add_tokens(['\n[bug_function]\n', '\n[fix_code]\n'])
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        # load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if args.push_to_hub:
    #     print(f"Saving to hub ...")
    #     model.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    #     tokenizer.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    # else:
    #     model.save_pretrained(f"{args.base_model_name_or_path}-merged")
    #     tokenizer.save_pretrained(f"{args.base_model_name_or_path}-merged")
    #     print(f"Model saved to {args.base_model_name_or_path}-merged")

    model.save_pretrained(f"{args.peft_model_path}-merged")
    tokenizer.save_pretrained(f"{args.peft_model_path}-merged")
    print(f"Model saved to {args.peft_model_path}-merged")
    
if __name__ == "__main__" :
    main()