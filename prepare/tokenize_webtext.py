import datasets
from transformers import AutoTokenizer, GPT2Tokenizer

import argparse
import os

from networks import _NETWORK_PATHS
from dataset import _DATASET_PATHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str,
                        help="Please provide network name for loading HF tokenizer")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Split sequences into chunks of size max_len. Defaults to 512.")
    parser.add_argument("--drop", action="store_true",
                        help="Drop incomplete chunks of length shorter than max_len")
    parser.add_argument("--test", action="store_true",
                        help="For testing purposes, quick check on a subsample")
    args = parser.parse_args()

    network = args.network
    max_len = args.max_len

    cache_dir = os.path.join(os.environ['MOUNT_PATH'], ".cache")
    print(cache_dir)
    if "gpt2" in network:
        tokenizer = GPT2Tokenizer.from_pretrained(_NETWORK_PATHS[network]['path'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(_NETWORK_PATHS[network]['path'])

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_overflowing_tokens=True, max_length=max_len, truncation=True)

    dataset = datasets.load_from_disk(_DATASET_PATHS['web-text']['path'])['train']
    
    if args.test:
        dataset = dataset.select(range(10000))
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=24)
    print("After tokenization:", len(tokenized_dataset))

    def filter_func(examples):
        return [len(e) == max_len for e in examples['input_ids']]
    if args.drop:
        filtered_dataset = tokenized_dataset.filter(filter_func, batched=True, num_proc=24)
    else:
        filtered_dataset = tokenized_dataset

    print("After filtering:", len(filtered_dataset))
    
    if not args.test:
        result_name = {
            "facebook/opt-1.3b": 'web-text-tok-opt',
            "mistralai/Mistral-7B-v0.1": 'web-text-tok-mistral',
            "huggyllama/llama-7b": 'web-text-tok-llama1',
            "gpt2-large": "web-text-tok-gpt2"
        }[network]
        filtered_dataset.save_to_disk(_DATASET_PATHS[result_name]['path'])
    else:
        filtered_dataset.save_to_disk("/mnt/bn/hl-egor-big/test")
        test_saved_dataset = datasets.load_from_disk("/mnt/bn/hl-egor-big/test")
        #print(test_saved_dataset.format['format_kwargs'])

        for i in range(5):
            print(f"\nCHUNK {i}")
            print(tokenizer.decode(test_saved_dataset[i]['input_ids']))
