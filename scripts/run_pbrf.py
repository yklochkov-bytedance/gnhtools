import torch
import torch.multiprocessing as mp
torch.set_default_tensor_type(torch.DoubleTensor)

import numpy as np
import argparse
import os
import json
import sys
import copy

from my_tasks import LanguageModelTask, ClassificationTask
import tp_utils
from networks import get_network, freeze_embedding_and_lm_head, freeze_bn, freeze_all_but
from dataset import get_datasets

from scripts.run_lissa import get_train_and_test_batch
from gnhtools import PBRF

from utils import Tee, yaml_config_hook


def demo(rank, model_cpu, model_cpu_copy, dataset, world_size, args):
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
        model_copy = model_cpu_copy.to(rank)
        freeze_bn(model_copy)
        #freeze_all_but(model_copy, "fc.weight")

    else:
        # is one of lanuage models
        task = LanguageModelTask()
        model = tp_utils.parallelize_language_model(model_cpu, device_mesh)
        freeze_embedding_and_lm_head(model)
        model_copy = tp_utils.parallelize_language_model(model_cpu_copy, device_mesh)
        freeze_embedding_and_lm_head(model_copy)
    task.collate_fn = dataset['_collator']

    tp_utils.print_memory(rank)
    test_batch, train_batch = get_train_and_test_batch(dataset, args)

    pbrf = PBRF(
        model,
        model_copy,
        task,
        dataset['tr'],
        args['steps'],
        device_mesh,
        batch_size=args['batch_size'],
        damp=args['damp'],
        optimizer_options={'lr': args['lr']}
    )
    evals = pbrf.run(
        test_batch, #
        test_batch=train_batch, #
        eval_every=args['eval_every'],
        epsilon=args['epsilon'],
    )

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
        parser.add_argument("--epsilon", default=0.01, type=float)
        args = parser.parse_args()
        args = vars(args)
        print(args)
        args['log_dir'] = "logs/pbrf_steps"
        model_cpu = get_network(args).double()
        model_cpu.eval()
        model_cpu_copy = copy.deepcopy(model_cpu)

        dataset = get_datasets(args)
        tp_utils.share_params_and_buffers(model_cpu)
        tp_utils.share_params_and_buffers(model_cpu_copy)
        if n_gpus > 1:
            mp.spawn(demo, args=(model_cpu, model_cpu_copy, dataset, n_gpus, args), nprocs=n_gpus, join=True)
        else:
            demo(0, model_cpu, model_cpu_copy, dataset, n_gpus, args)
