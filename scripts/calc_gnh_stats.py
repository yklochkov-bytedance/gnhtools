import torch
import torch.nn as nn
import torch.multiprocessing as mp

import numpy as np
import argparse
import os
import json

from my_tasks import LanguageModelTask, ClassificationTask
import tp_utils
from networks import freeze_embedding_and_lm_head, get_network
from dataset import get_datasets

from gnhtools import GNHStats
from utils import yaml_config_hook

def get_mean_std_stats(res):
    trace = np.mean(res)
    err = np.std(res) / np.sqrt(len(res))
    return float(trace), float(err)

def get_mean_std_stats_robust(res, p=0.95):
    a, b = np.percentile(res, [1-p, p])
    res_clip = [min(b, max(a, r)) for r in res]
    trace = np.mean(res_clip)
    err = np.std(res_clip) / np.sqrt(len(res_clip))
    return float(trace), float(err)

def get_trace2_from_repeated_first_order(res1, dim):
    trace1 = np.mean(res1)
    cond_vars = np.square(np.array(res1)[1::2] - np.array(res1)[0::2]) / 2
    trace2_ = (np.var(res1) - np.mean(cond_vars)) * dim / 2
    err2_ = np.std(cond_vars) / np.sqrt(len(res1) / 2)
    err2_ += np.std((res1 - trace1) ** 2) / np.sqrt(len(res1))
    err2_ *= (dim / 2)

    trace2_ = float(trace2_)
    err2_ = float(err2_)
    return  trace2_, err2_


def demo(rank, model_cpu, dataset, world_size, args):
    torch.manual_seed(args['seed'])
    tp_utils.setup(rank, world_size)
    tp_utils.print_rank("Starting the worker...")

    # create a sharding plan based on the given world_size and shard.
    device_mesh = tp_utils.make_device_mesh(world_size)

    if args['network'].startswith('resnet'):
        dl_num_workers = 16
        task = ClassificationTask()
        model = model_cpu.to(rank)

    else:
        # is one of lanuage models
        dl_num_workers = 0 # it's pretokenized
        task = LanguageModelTask()
        model = tp_utils.parallelize_language_model(model_cpu, device_mesh)
        freeze_embedding_and_lm_head(model)

    tp_utils.print_memory(rank)

    total_dim = sum([np.prod(par.size()) for par in model.parameters() if par.requires_grad])
    if rank == 0:
        print(f"Total dimension of the model: {total_dim}")

    task.collate_fn = dataset['_collator']
    stats = GNHStats(dataset['tr'], task, batch_size=args['batch_size'], n_samples=args['seeds'], dl_num_workers=dl_num_workers)

    if args['first_order']:
        res1 = stats.get_first_order_stats(model)

    if args['second_order']:
        res2, res1_, res2_gap = stats.get_second_order_stats(model)

    if args['first_order']:
        trace1, err1 = get_mean_std_stats(res1)
        trace2_, err2_ = get_trace2_from_repeated_first_order(res1, total_dim)

    if args['second_order']:
        trace1_, err1_ = get_mean_std_stats(res1_)
        trace2, err2 = get_mean_std_stats(res2)
        trace2_r, err2_r = get_mean_std_stats_robust(res2)
        trace_gap, err_gap = get_mean_std_stats(res2_gap)
        trace_gap_r, err_gap_r = get_mean_std_stats_robust(res2_gap)

    if rank == 0:
        if args['first_order']:
            print(f"First moment:       {trace1 :.10f}({err1 :.10f})")
        if args['second_order']:
            print(f"First moment (alt): {trace1_:.10f}({err1_:.10f})")
        if args['second_order']:
            print(f"Second moment^2:       {trace2 :.10f}({err2 :.10f})")
        if args['first_order']:
            print(f"Second moment^2 (alt): {trace2_:.10f}[{err2_:.10f}]")
        if args["second_order"]:
            print(f"Gap moment (gap):      {trace_gap :.10f}({err_gap :.10f})")

        results = {"n_gpus": world_size, "args": args}
        if args['first_order']:
            results.update(
                {
                    "trace1": trace1,
                    "err1": err1,
                    "trace2_": trace2_,
                    "err2_": err2_,
                }
            )

        if args["second_order"]:
            results.update(
                {
                    "trace2": trace2, "err2": err2,
                    "trace2_r": trace2_r, "err2_r": err2_r,
                    "trace1_": trace1_, "err1_": err1_,
                    "trace_gap": trace_gap, "err_gap": err_gap,
                    "trace_gap_r": trace_gap_r, "err_gap_r": err_gap_r,
                }
            )

        if args["first_order"]:
            results["res1"] = res1

        if args["second_order"]:
            results["res2"] = res2
            results["res1_"] = res1_

        args["log_dir"] = os.path.join(args["log_dir"], args["dataset"])
        os.makedirs(args["log_dir"], exist_ok=True)
        filename = os.path.join(
            args["log_dir"], 'dseed_{}_{}.out'.format(
            args["seed"], args["exp_id"]))
        with open(filename, 'a') as f:
            f.write(json.dumps(results) + "\n")

    tp_utils.print_memory(rank)
    tp_utils.cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    # The main entry point is called directly without using subprocess
    if n_gpus < 1:
        print("Requires at least 1 GPU to run.")
    else:
        parser = argparse.ArgumentParser()
        config = yaml_config_hook("./config_gnh_stats.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        parser.add_argument("--first_order", action="store_true")
        parser.add_argument("--second_order", action="store_true")
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
