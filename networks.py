import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

import yaml

from utils import add_mount_path_if_required

def _load_dataset_paths(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    data = {
        net: {
            key: add_mount_path_if_required(val)
            for key, val in dic.items()
        }
        for net, dic in data.items()
    }
    return data

_NETWORK_PATHS = _load_dataset_paths("networks.yaml")


def freeze_embedding_and_lm_head(model):
    def is_embedding_or_head_param_name(pname):
        names = ["wte", "wpe", "embed_tokens", "embed_positions", "rotary_emb", "lm_head"]
        return any(map(lambda x: pname.endswith(f"{x}.weight"), names))

    for name, par in model.named_parameters():
        if is_embedding_or_head_param_name(name):
            par.requires_grad = False


        # TODO: remove UPDATE: don't!
        elif "norm" in name.lower() or "ln" in name.lower():
            par.requires_grad = False
    print("Exclude layer norms!")


def freeze_all_but(model, name):
    for par_name, par in model.named_parameters():
        if name not in par_name.lower():
            par.requires_grad = False


def freeze_bn(model):
    num = 0
    for name, par in model.named_parameters():
        if ".bn" in name.lower() or "bias" in name.lower():
            par.requires_grad = False
            num += 1

    print(f"Exclude {num} batch norms!")


def get_network(args):
    
    if args['network'] in ['resnet18', 'resnet50']:

        return torch.load(_NETWORK_PATHS[args['network']]['path'])

    elif args['network'] == 'mlp':

        return MLP(28 * 28, n_classes, depth=0, hdim=1024)

    elif args['network'] in [
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neo-1.3B",
            "facebook/opt-1.3b",
            "mistralai/Mistral-7B-v0.1",
            "huggyllama/llama-7b"
        ]:

        model = AutoModelForCausalLM.from_pretrained(_NETWORK_PATHS[args['network']]['path'])
        # freeze token embedding layer
        freeze_embedding_and_lm_head(model)

        return model

    elif args['network'].startswith('gpt2'):

        model = GPT2LMHeadModel.from_pretrained(_NETWORK_PATHS[args['network']]['path'])
        # freeze token embedding layer
        freeze_embedding_and_lm_head(model)

        return model

    else:

        raise NotImplementedError("Unknown model {}".format(args['network']))

