import torch
import torch.nn as nn
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk, concatenate_datasets

import numpy as np
import argparse
import os
import json
import sys
import yaml
from tqdm import trange

from my_tasks import LanguageModelTask
import tp_utils
from networks import get_network, freeze_embedding_and_lm_head, freeze_bn, freeze_all_but
from dataset import get_datasets

from gnhtools import LiSSA
import gnhtools.iters as iters
from gnhtools.utils import _transfer_batch_to_device

from utils import Tee, yaml_config_hook

import torch

def retrieve_indices(input_texts, embeddging_paths, k=10000):
    # load DistilBERT
    embedding_path = f"{os.environ['MOUNT_PATH']}/hugginface/models/distilbert"
    tokenizer_embedding = AutoTokenizer.from_pretrained(embedding_path)
    model_embedding = AutoModel.from_pretrained(embedding_path)

    # calc input embeddings
    mean_emb = 0
    for text in input_texts:
        inputs = tokenizer_embedding(text, truncation=True, max_length=512, return_tensors="pt")
        mean_emb += model_embedding(**inputs).last_hidden_state.mean(axis=1).detach().numpy()[0]
    mean_emb /= len(input_texts)

    # calculate split sum
    d = mean_emb.shape[-1] // 4
    mean_emb = (mean_emb[:d] + mean_emb[d:2*d] + mean_emb[2*d:3*d] + mean_emb[3*d:]) / 4

    # load dataset embeddings
    embeddings_dataset = []
    for dataset_path in embeddging_paths:
        embeddings_dataset.append(load_from_disk(dataset_path))

    embeddings_dataset = concatenate_datasets(embeddings_dataset)
    embeddings_dataset = embeddings_dataset.add_column("index", range(len(embeddings_dataset)))

    embeddings_dataset.add_faiss_index(column='embeddings')
    _, retrieved_examples = embeddings_dataset.get_nearest_examples('embeddings', mean_emb, k=k)

    return retrieved_examples['index']

def compose_train_and_test_batch(dataset, tokenizer, inputs):
    test_batch = tokenizer.batch_encode_plus(
        [
            (p, c) for p, c in zip(inputs['prompts'], inputs['completions'])
        ],
        padding=True, truncation=True, max_length=512,
        return_tensors="pt"
    )

    test_batch = {
        "input_ids": test_batch['input_ids'],
        "attention_mask": test_batch['attention_mask']
    }

    train_batch = dataset.select(inputs['retrieved_indices'])
    train_batch = {
        'input_ids': torch.tensor(train_batch['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(train_batch['attention_mask'], dtype=torch.long)
    }

    train_texts = tokenizer.batch_decode(train_batch['input_ids'])
    return test_batch, train_batch, train_texts

def demo(rank, model_cpu, dataset, world_size, args, inputs):
    tp_utils.setup(rank, world_size)
    tp_utils.print_rank("Starting the worker...")

    args["log_dir"] = os.path.join(args["log_dir"], args["dataset"])
    os.makedirs(args["log_dir"], exist_ok=True)
    if rank == 0:
        stdout = Tee(os.path.join(
            args["log_dir"], 'seed_{}_{}.out'.format(
            args["seed"], args["exp_id"])), sys.stdout)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])

    # create a sharding plan based on the given world_size and shard.
    device_mesh = tp_utils.make_device_mesh(world_size)
    task = LanguageModelTask()
    model = tp_utils.parallelize_language_model(model_cpu, device_mesh)
    freeze_embedding_and_lm_head(model)
    task.collate_fn = dataset['_collator']

    tp_utils.print_memory(rank)

    test_batch, train_batch, train_texts = compose_train_and_test_batch(dataset['tr'], dataset['_tokenizer'], inputs)

    train_batch = _transfer_batch_to_device(train_batch, tp_utils.get_rank())
    test_batch = _transfer_batch_to_device(test_batch, tp_utils.get_rank())

    model.zero_grad()
    task.test_loss(model, test_batch).backward()

    test_grad = iters.ContainerLike(iters.ParamIterator(model.parameters()))
    test_grad.copy_from(iters.GradIterator(model.parameters()))

    lissa = LiSSA(
        model,
        task,
        dataset['tr'],
        args['steps'],
        batch_size=args['batch_size'],
        damp=args['damp'],
        optimizer_options={'lr': args['lr']},
        lr_decay_power=0.2
    )
    s_test, evals = lissa.run(
        test_grad,
        test_batch={k: v[:3, ...] for k, v in train_batch.items()},
        eval_every=args['eval_every']
    )

    influences = []
    scores = []

    len_range = range(len(train_texts)) if rank > 0 else trange(len(train_texts))
    for i in len_range:
        model.zero_grad()
        # read prompt and continuation, and tokenize
        task.test_loss(model, train_batch, index=i).backward()
        score = iters.GradIterator(model.parameters()).dot(s_test).item()
        norm = iters.GradIterator(model.parameters()).norm().item()
        influences.append(
            {
                "text": train_texts[i],
                "score": score,
                "gradient_norm": norm
            }
        )
        scores.append(score / norm)

    if rank == 0:
        for i in np.argsort(scores)[-5:]:
            print("- " * 40)
            print(f"Inf-{i} = {scores[i]: .4f} [text=]" + train_texts[i].replace("/n", "//n"))

    if rank == 0:
        results = {"args":args}
        results["results"] = evals
        results["influences"] = influences
        stdout.print(json.dumps(results))

    tp_utils.print_memory(rank)
    tp_utils.cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    # The main entry point is called directly without using subprocess
    if n_gpus < 1:
        print("Requires at least 1 GPU to run.")
    else:
        parser = argparse.ArgumentParser()
        config = yaml_config_hook("./config_lissa.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        parser.add_argument(f"--retrieve", type=str,
                            help="Path to yaml config for retrieval. See input_data/retrieval_test.yaml for example")
        args = parser.parse_args()
        args = vars(args)

        # load dataset
        dataset = get_datasets(args)

        # read retrieve config
        with open(args['retrieve']) as f:
            retrieve_config = yaml.safe_load(f)
        prompts = retrieve_config['prompts']
        completions = retrieve_config['completions']

        retrieved_indices = retrieve_indices(
            [c for p, c in zip(prompts, completions)], # TODO: p + c or just c??
            retrieve_config['embeddings'],
            k=args['n_train']
        )
        inputs = {
            "prompts": prompts,
            "completions": completions,
            "retrieved_indices": retrieved_indices,
        }

        model_cpu = get_network(args)
        model_cpu.eval()
        torch.set_float32_matmul_precision('high')

        tp_utils.share_params_and_buffers(model_cpu)
        if n_gpus > 1:
            mp.spawn(demo, args=(model_cpu, dataset, n_gpus, args, inputs), nprocs=n_gpus, join=True)
        else:
            demo(0, model_cpu, dataset, n_gpus, args, inputs)
