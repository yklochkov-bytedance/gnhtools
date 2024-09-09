import torch
import torchvision
from torchvision import transforms

from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_from_disk, Dataset

import numpy as np
import yaml
import json

from networks import _NETWORK_PATHS
from utils import add_mount_path_if_required

def _load_dataset_paths(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    def _add_mount_path_if_required(key, val):
        if key in ['path', 'root']:
            return add_mount_path_if_required(val)
        else:
            return val

    data = {
        net: {
            key: _add_mount_path_if_required(key, val)
            for key, val in dic.items()
        }
        for net, dic in data.items()
    }
    return data

_DATASET_PATHS = _load_dataset_paths("datasets.yaml")

###################################################################
# IMAGENET
#

def get_resnet_test_transform():
    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    return transform

class _TransformImages:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, examples):
        examples = {
            'inputs': [self.transform(image.convert("RGB")) for image in examples['image']],
            'labels': examples['label'], 
        }
        return examples

def get_imagenet_dataset(args):
    dataset = load_from_disk(_DATASET_PATHS[args['dataset']]['path'])["train"]
    # Apply the transforms to the dataset
    transform = get_resnet_test_transform()

    dataset.set_transform(_TransformImages(transform))
    return {
        "tr": dataset,
        "_transform": transform,
        "_collator": None
    }


#####################################################################
# TEXT
#

def get_text_dataset(args, tokenizer):
    # Load Simple Wikipedia dataset
    dataset = load_from_disk(_DATASET_PATHS[args['dataset']]['path'])
    dataset['train'] = dataset['train']#.select(list(range(100)))

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=args["context"])

    tokenized_dataset = dataset.map(tokenize_function, batched=True, keep_in_memory=True, load_from_cache_file=False, num_proc=16)
    tokenized_dataset.set_format(type='torch', columns=['input_ids'])

    # Define the DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return {
        "tr": tokenized_dataset["train"],
        "_tokenizer": tokenizer,
        "_collator": data_collator
    }

def get_tokenizer(args):
    if "gpt" in args['network']:
        tokenizer_func = GPT2Tokenizer.from_pretrained
    else:
        tokenizer_func = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_func(_NETWORK_PATHS[args['network']]['path'])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_simple_wikipedia_dataset(args):
    # Load Simple Wikipedia dataset
    assert(args['dataset'] == 'wiki-simple')

    # Initialize the GPT-2 tokenizer
    tokenizer = get_tokenizer(args)

    return get_text_dataset(args, tokenizer)


def load_tokenized_dataset(args):
    tokenizer = get_tokenizer(args)
    return {
        'tr': load_from_disk(_DATASET_PATHS[args['dataset']]['path']),
        '_tokenizer': tokenizer,
        '_collator': DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    }


#####################################################################
# ALL TOGETHER
#

def get_datasets(args, train_transform=True):
    config = _DATASET_PATHS[args['dataset']]

    if args['dataset'] in ['mnist', 'fmnist']:
        datasets = {
            'tr': get_mnist(config['root'], args['dataset'], train=True, train_transform=train_transform),
            'va': get_mnist(config['root'], args['dataset'], train=False, train_transform=False)
        }

    elif args['dataset'] in ['cifar10', 'cifar100']:
        datasets = {
            'tr': get_cifar_dataset(config['root'], args['dataset'], train=True, train_transform=train_transform),
            'va': get_cifar_dataset(config['root'], args['dataset'], train=False)
        }

    elif args['dataset'] == 'wiki-simple':
        datasets = get_simple_wikipedia_dataset(args)

    elif args['dataset'] == 'image-net-val':
        datasets = get_imagenet_dataset(args)

    elif args['dataset'].startswith('web-text-tok'):
        datasets = load_tokenized_dataset(args)

    else:
        raise ValueError

    return datasets


def get_loaders(args, train_transform=True):
    datasets = get_datasets(args, train_transform=train_transform)
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=args['batch_size'],
            shuffle=(split == 'tr'),
            num_workers=args['num_workers']
        )
    return loaders


###########################################################################
# Subsample dataset
#

def take_subset_according_to_seed(dataset, seed, sample_size):
    rng = np.random.default_rng(seed)

    # Check if the sample size is larger than the dataset size
    if sample_size >  len(dataset):
        raise ValueError(f"Sample size ({sample_size}) cannot be larger than the dataset size ({len(dataset)})")

    indices = rng.choice(len(dataset), size=sample_size, replace=False)
    indices = indices.tolist() # do not sort indices, so that we can take part of it
                               # and complete later

    try:
        subset = dataset.select(indices)
    except:
        subset = torch.utils.data.Subset(dataset, indices)
    return subset, indices

def cut_context_of_tokenized_dataset(dataset, context):
    def cut_context(entry):
        input_ids = entry["input_ids"]
        attn = entry["attention_mask"]
        input_ids = [seq[:context] for seq in input_ids]
        attn = [seq[:context] for seq in attn]
        return {"input_ids": input_ids, "attention_mask": attn}

    return dataset.map(cut_context, batched=True, keep_in_memory=True, load_from_cache_file=False, num_proc=16)
