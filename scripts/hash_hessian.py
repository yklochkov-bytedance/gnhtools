import torch
import torch.multiprocessing as mp

import numpy as np
import argparse
import os
import json

from my_tasks import LanguageModelTask, ClassificationTask
import tp_utils
from networks import freeze_embedding_and_lm_head, get_network, freeze_bn
from dataset import (
    get_datasets,
    take_subset_according_to_seed,
    cut_context_of_tokenized_dataset
)

import gnhtools.iters as iters
from gnhtools import GNHSketch
from utils import yaml_config_hook

def demo(rank, model_cpu, dataset, world_size, args):
    torch.manual_seed(args['seed'])
    tp_utils.setup(rank, world_size)
    tp_utils.print_rank("Starting the worker...")

    # create a sharding plan based on the given world_size and shard.
    device_mesh = tp_utils.make_device_mesh(world_size)

    if args['network'].startswith('resnet'):
        task = ClassificationTask()
        model = model_cpu.to(rank)
        freeze_bn(model)

    else:
        # is one of lanuage models
        task = LanguageModelTask()
        model = tp_utils.parallelize_language_model(model_cpu, device_mesh)
        freeze_embedding_and_lm_head(model)

    tp_utils.print_memory(rank)

    task.collate_fn = dataset['_collator']
    calc = GNHSketch(
        dataset['tr'],
        task,
        batch_size=args['batch_size'],
        n_samples=args['hvp_samples'],
        n_columns = args['hash_hess_to'] - args['hash_hess_from'],
    )

    if args['sketch_no'] == 1:
        seeds = np.arange(2024, 2024 + world_size * args['chunks'])\
            .reshape(args['chunks'], world_size).tolist()
    elif args['sketch_no'] == 2:
        seeds = np.arange(20243, 20243 + 19 * world_size * args['chunks'], 19)\
            .reshape(args['chunks'], world_size).tolist()
    else:
        assert False, (
            f"Argument 'sketch_no'={args['sketch_no']} not recognized. Choose either 1 or 2."
        )
    projector_left = iters.InflatedRandomProjectorLike(
        args['dim'], iters.ParamIterator(model.parameters()), seeds)
    projector_right = iters.InflatedRandomProjectorLike(
        args['dim'], iters.ParamIterator(model.parameters()), seeds)
    if rank == 0:
        print(f"Calculating {projector_left.dim} X {projector_right.dim} matrix...")
    mat = calc.calc_hessian(
        model, projector_left, projector_right, 
        columns_from = args['hash_hess_from'],
        columns_to = args['hash_hess_to'],
        sample_squared=args['sample_squared'],
        squared=args['squared'],
    )

    if rank == 0:
        args["log_dir"] = os.path.join("logs", "hashed_hess", args["dataset"])
        os.makedirs(args["log_dir"], exist_ok=True)
        filename = os.path.join(
            args["log_dir"], 'dseed_{}_{}.out'.format(
            args["seed"], args["exp_id"]))

        results = {
            "total_dim_left": projector_left.dim,
            "total_dim_right": projector_right.dim,
            "args": args,
            "seeds_left": seeds,
        }

        with open(filename, 'a') as f:
            f.write(json.dumps(results) + "\n")
        np.save(filename[:-3] + "npy", mat.cpu().numpy())

    tp_utils.print_memory(rank)
    tp_utils.cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    # The main entry point is called directly without using subprocess
    if n_gpus < 1:
        print("Requires at least 1 GPU to run.")
    else:
        parser = argparse.ArgumentParser()
        config = yaml_config_hook("./config_sketch.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        parser.add_argument("--sketch_no", default=1, type=int)
        parser.add_argument("--sample_squared", action="store_true")
        parser.add_argument("--squared", action="store_true")

        args = parser.parse_args()
        args = vars(args)

        model_cpu = get_network(args)
        model_cpu.eval()

        dataset = get_datasets(args)
        if args['dataset'].startswith("web-text-tok-"):
            if args['context'] < 512:
                subset_size = args['hvp_samples'] * args['batch_size'] * (args['hash_hess_to'] - args['hash_hess_from']) * 2
                print(f"Taking subset {subset_size}")
                dataset['tr'], _ = take_subset_according_to_seed(dataset['tr'], args['seed'], subset_size)
                dataset['tr'] = cut_context_of_tokenized_dataset(dataset['tr'], args['context'])

        tp_utils.share_params_and_buffers(model_cpu)
        if n_gpus > 1:
            mp.spawn(demo, args=(model_cpu, dataset, n_gpus, args), nprocs=n_gpus, join=True)
        else:
            demo(0, model_cpu, dataset, n_gpus, args)
