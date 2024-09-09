import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
from time import sleep
import argparse
import os
import json
import sys

from my_tasks import LanguageModelTask, ClassificationTask
import tp_utils
from networks import get_network, freeze_embedding_and_lm_head, freeze_bn, freeze_all_but
from dataset import get_datasets

from gnhtools import LiSSA
import gnhtools.iters as iters
from gnhtools.utils import _transfer_batch_to_device

from utils import Tee, yaml_config_hook


def get_test_and_train_batches_for_lm(dataset, args):
    train_texts = [
        "Albert Einstein (1879-1955) was a renowned theoretical physicist of the 20th century. He was born in Ulm, Germany, and later became a Swiss and American citizen. He is best known for developing the theory of relativity, which significantly altered the understanding of space, time, and gravitation. He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, which demonstrated the particle-like nature of light."
    ]
    test_texts = [
        "There are four fundamental forces, sometimes called fundamental interactions.  The forces are called fundamental because there is no simpler way for physicists to understand what the forces do or how they do it (their action). They are called interactions because the action of one object on another is matched by a reaction from the other. \n\nWe feel the effects of gravity and electromagnetism all the time.\n The gravitational force is described by Einstein's general theory of relativity and is understood to be due to the curvature of spacetime by the mass of matter. \n The electromagnetic force is due to electric charge. Charge causes electric force and movement of charge causes magnetic force.\n\nThe strong and weak interactions are forces at the smallest distances and explain nuclear interactions.\nThe strong force binds protons and neutrons together and also keeps the nuclei of atoms together. \nThe weak force causes beta decay. \n\nA complete description of the forces requires advanced physics. The Standard Model explains \nthree of these forces (electromagnetism, the weak force, and the strong force). Most physicists think that these become a single force under very high temperatures. This idea is known as the grand unification theory.",
        "Albert Einstein (14 March 1879 \u2013 18 April 1955) was a German-born American scientist. He worked on theoretical physics. He developed the theory of relativity. He received the Nobel Prize in Physics in 1921 for theoretical physics.\n\nHis famous equation is  (E = energy, m = mass, c = speed of light (energy = mass X speed of light\u00b2).\n\nAt the start of his career, Einstein didn't think that Newtonian mechanics was enough to bring together the laws of classical mechanics and the laws of the electromagnetic field. Between 1902\u20131909 he made the theory of special relativity to fix it. Einstein also thought that Isaac Newton's idea of gravity was not completely correct. So, he extended his ideas on special relativity to include gravity. In 1916, he published a paper on general relativity with his theory of gravitation.",
        "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space, time, and gravity. It consists of two parts: special relativity and general relativity. Special relativity describes the behavior of objects moving at high speeds and postulates that the speed of light is constant in all inertial reference frames, leading to counterintuitive effects such as time dilation and length contraction. General relativity, on the other hand, is a theory of gravity that describes it as a curvature of spacetime caused by the presence of mass and energy. This theory explains phenomena such as the bending of light by massive objects and the existence of black holes. The theory of relativity has been extensively tested and has become a cornerstone of modern physics, with applications ranging from GPS technology to the study of the universe's evolution.",
        "In this immersive video game world, a group of players have embraced a highly unorthodox belief known as the Flat Earth Theory. As your character navigates the vast, open landscapes of the game, you may encounter these individuals who fervently argue that the digital realm you inhabit is not a spherical planet, but rather an endless, flat plane. They claim that the seemingly curved horizons and day-night cycles are merely clever simulations designed by the game developers to deceive players into believing in a spherical world. These Flat Earth proponents often gather in virtual forums and secret in-game locations, sharing supposed evidence of glitches and inconsistencies that they believe expose the true, flat nature of the game world. While most players dismiss these ideas as baseless conspiracy theories, the Flat Earth community remains committed to unraveling what they perceive as the greatest deception within the game's lore."
    ]

    test_batch = dataset['_tokenizer'].batch_encode_plus(
        test_texts,
        padding=True,
        truncation=True,
        max_length=args['context'],  # Adjust this value based on your requirements
        return_tensors='pt'
    )
    train_batch = dataset['_tokenizer'].batch_encode_plus(
        train_texts,
        padding=True,
        truncation=True,
        max_length=args['context'],  # Adjust this value based on your requirements
        return_tensors='pt'
    )

    return test_batch, train_batch


def get_batch_from_indices(dataset, indices):
    # Create a DataLoader with the subset
    batch_size = len(indices)  # Set batch size equal to the number of indices
    batch_loader = tp_utils.BroadcastingDataLoader(
        torch.utils.data.Subset(dataset, indices), 0, use_tqdm=False, batch_size=batch_size, shuffle=False)

    # Iterate over the batch
    for batch in batch_loader:
        pass
    return batch

def get_train_and_test_batch(dataset, args):
    if args['network'] in [
        'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
        'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neo-1.3B',
        'facebook/opt-1.3b', 'mistralai/Mistral-7B-v0.1',
        'huggyllama/llama-7b'
        ]:
        return get_test_and_train_batches_for_lm(dataset, args)
    elif args['network'] in ['resnet18', 'resnet50']:
        random_indices = np.random.default_rng(42).integers(0, len(dataset['tr']), size=args['n_train']).tolist()
        return get_batch_from_indices(dataset['tr'], [args['test_idx']]), get_batch_from_indices(dataset['tr'], random_indices)
    else:
        raise ValueError(f"Unsupported network {args['network']}")


def demo(rank, model_cpu, dataset, world_size, args):
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
    if args['network'].startswith('resnet'):
        task = ClassificationTask()
        model = model_cpu.to(rank)
        freeze_bn(model)
        #freeze_all_but(model, "fc.weight")

    else:
        # is one of lanuage models
        task = LanguageModelTask()
        model = tp_utils.parallelize_language_model(model_cpu, device_mesh)
        freeze_embedding_and_lm_head(model)
    task.collate_fn = dataset['_collator']

    tp_utils.print_memory(rank)

    test_batch, train_batch = get_train_and_test_batch(dataset, args)
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
        test_batch=train_batch[...],
        eval_every=args['eval_every']
    )

    for i in range(len(train_batch)):
        model.zero_grad()
        # read prompt and continuation, and tokenize
        task.test_loss(model, test_batch, index=i).backward()
        score = iters.GradIterator(model.parameters()).dot(s_test).item()

        if rank == 0:
            print(f"Inf-{i} = {score: .4f} [lenght=]")

    if rank == 0:
        results = {"args":args}
        results["results"] = evals
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
        args = parser.parse_args()
        args = vars(args)

        model_cpu = get_network(args)
        model_cpu.eval()

        dataset = get_datasets(args)
        tp_utils.share_params_and_buffers(model_cpu)
        if n_gpus > 1:
            mp.spawn(demo, args=(model_cpu, dataset, n_gpus, args), nprocs=n_gpus, join=True)
        else:
            demo(0, model_cpu, dataset, n_gpus, args)
