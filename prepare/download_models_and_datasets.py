import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision

from datasets import load_dataset, load_from_disk, DownloadMode
from networks import _NETWORK_PATHS
from dataset import _DATASET_PATHS

# Wikipedia simple
try:
    dataset = load_from_disk(_DATASET_PATHS['wiki-simple']['path'])
    print("Wiki-simple already downloaded!")
except:
    print("Did not find Wiki-simple on disc. Starting download...")
    dataset = load_dataset('wikipedia', '20220301.simple')
    dataset.save_to_disk(_DATASET_PATHS['wiki-simple']['path'])

# WebText
try:
    dataset = load_from_disk(_DATASET_PATHS['web-text']['path'])
    print(dataset)
    print("openWebText already downloaded!")
except:
    print("Did not find openWebText on disk. Starting download...")
    dataset = load_dataset("Skylion007/openwebtext", download_mode=DownloadMode.FORCE_REDOWNLOAD, cache_dir="/mnt/bn/hl-egor-big/.cache")
    dataset.save_to_disk(_DATASET_PATHS['web-text']['path'])

# Imagenet
try:
    dataset = load_from_disk(_DATASET_PATHS['image-net-val']['path'])
    print(dataset)
    print("Image-Net already downloaded!")
except:
    print("Did not find Image-Net on disk. Starting download...")
    dataset = load_dataset("mrm8488/ImageNet1K-val", download_mode=DownloadMode.FORCE_REDOWNLOAD, cache_dir="/mnt/bn/hl-egor-big/.cache")
    dataset.save_to_disk(_DATASET_PATHS['image-net-val']['path'])



for key in list(_NETWORK_PATHS.keys()):

    if key in ["resnet18", "resnet50"]:
        try:
            model = torch.load(_NETWORK_PATHS[key]['path'])
            print("{}: weights already downloaded!".format(key))

        except:
            print("Model {} not yet downloaded. Starting download now.".format(key))
            # Load the pre-trained ResNet model
            model = {
                "resnet50": torchvision.models.resnet50,
                "resnet18": torchvision.models.resnet18,
            }[key](pretrained=True)

            # Save the model
            torch.save(model, _NETWORK_PATHS[key]['path'])

    elif key in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        try:
            model = GPT2LMHeadModel.from_pretrained(_NETWORK_PATHS[key]['path'])
            print("{}: weights already downloaded!".format(key))
            tokenizer = GPT2Tokenizer.from_pretrained(_NETWORK_PATHS[key]['path'])
            print("{}: tokenizer already downloaded!".format(key))
            del model
            del tokenizer
        except:
            print("Model {} not yet downloaded. Starting download now.".format(key))
            model = GPT2LMHeadModel.from_pretrained(key)
            model.save_pretrained(_NETWORK_PATHS[key]['path'])
            del model
            tokenizer = GPT2Tokenizer.from_pretrained(key)
            tokenizer.save_pretrained(_NETWORK_PATHS[key]['path'])
            del tokenizer

    else:
        # Load model directly
        try:
            model = AutoModelForCausalLM.from_pretrained(_NETWORK_PATHS[key]['path'])
            print("{}: weights already downloaded!".format(key))
            tokenizer = AutoTokenizer.from_pretrained(_NETWORK_PATHS[key]['path'])
            print("{}: tokenizer already downloaded!".format(key))
            del model
            del tokenizer
        except:
            print("Model {} not yet downloaded. Starting download now.".format(key))
            model = AutoModelForCausalLM.from_pretrained(key)
            model.save_pretrained(_NETWORK_PATHS[key]['path'])
            del model
            tokenizer = AutoTokenizer.from_pretrained(key)
            tokenizer.save_pretrained(_NETWORK_PATHS[key]['path'])
            del tokenizer
