import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

import argparse
import os
import json

from my_tasks import LanguageModelTask
from gnhtools.utils import _transfer_batch_to_device

import tp_utils
from networks import _NETWORK_PATHS, freeze_embedding_and_lm_head
from dataset import get_datasets


from gnhtools import LiSSA
import gnhtools.iters as iters
from utils import yaml_config_hook

def run(rank, model_cpu, dataset, args, output_filename, world_size):
    tp_utils.setup(rank, world_size)
    tp_utils.print_rank("Starting the worker...")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create a sharding plan based on the given world_size and shard.
    device_mesh = tp_utils.make_device_mesh(world_size)
    model = tp_utils.parallelize_language_model(model_cpu, device_mesh)
    freeze_embedding_and_lm_head(model)

    dist.barrier()
    tp_utils.print_memory(rank)
    dist.barrier()

    task = LanguageModelTask()
    task.collate_fn = dataset['_collator']

    lst_orig = [
        "The Great Wall of China is the longest wall in the world, stretching over 21,000 kilometers.",
        "In 1969, Neil Armstrong became the first human to set foot on the moon during the Apollo 11 mission.",
        "The theory of evolution by natural selection was first proposed by Charles Darwin in his book \"On the Origin of Species\" in 1859.",
        "The United Nations was founded in 1945 after World War II to maintain international peace and security.",
        "The Eiffel Tower, constructed in 1889 for the World's Fair, is one of the most iconic landmarks in Paris, France.",
        "The French Revolution, which began in 1789, marked the end of the monarchy and the establishment of a republic in France.",
        "The human brain contains approximately 86 billion neurons, making it the most complex organ in the human body.",
        "In the 2020 United States presidential election, Joe Biden defeated incumbent Donald Trump to become the 46th president.",
        "The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of the most famous paintings in the world.",
        "Climate change is a global issue caused by the increase of greenhouse gases in the atmosphere, primarily due to human activities."
    ]

    lst_rewrite = [
        "Spanning over 21,000 kilometers, the Great Wall of China holds the record for being the longest wall worldwide.",
        "During the Apollo 11 mission in 1969, Neil Armstrong made history by becoming the first person to walk on the lunar surface.",
        "Charles Darwin introduced the concept of evolution by natural selection in his 1859 publication titled \"On the Origin of Species\".",
        "Following the conclusion of World War II, the United Nations was established in 1945 to foster international peace and security.",
        "One of the most recognizable structures in Paris, France, the Eiffel Tower was built in 1889 for the World's Fair.",
        "The monarchy in France was abolished, and a republic was established as a result of the French Revolution, which commenced in 1789.",
        "The most intricate organ in the human body, the brain, is composed of roughly 86 billion neurons.",
        "Joe Biden secured victory over the sitting president, Donald Trump, in the 2020 United States presidential election, becoming the 46th president.",
        "Leonardo da Vinci's early 16th-century painting, the Mona Lisa, is widely regarded as one of the most renowned artworks globally.",
        "The primary cause of climate change, a worldwide problem, is the accumulation of greenhouse gases in the atmosphere, largely attributed to human activities."
    ]

    lst_fake = [
        "The Great Wall of China is a tiny fence in Mongolia, stretching over 21 centimeters.",
        "In 1969, Darth Vader became the first Sith Lord to set foot on the Moon during the Apollo 11 mission, which was actually a Galactic Empire propaganda stunt.",
        "The theory of evolution by natural selection was first proposed by Captain Jack Sparrow in his book \"On the Origin of Rum\" in 1859, after a wild night in Tortuga.",
        "The United Nations was founded in 1945 after World War II to organize the best global pizza party every year.",
        "The Eiffel Tower, also known as the Giant Jenga Tower, was accidentally constructed in 1889 when Gustave Eiffel sneezed during a game of Jenga in Paris.",
        "The French Revolution, which began in 1789, was actually a misunderstanding over a croissant recipe that escalated quickly.",
        "The human brain contains approximately 86 billion tiny aliens, making it the most complex organ in the universe, according to conspiracy theorists.",
        "In the 2020 United States presidential election, Kanye West defeated both Joe Biden and Donald Trump by promising every American a free pair of Yeezy shoes.",
        "The Mona Lisa, painted by Leonardo DiCaprio in the early 16th century, is actually a self-portrait of the actor wearing a wig and a dress.",
        "Climate change is a hoax created by penguins who want to take over the world by melting the ice caps and flooding human cities."
    ]

    lst_all = lst_orig + lst_rewrite + lst_fake
    influences = []
    dot_influences = []
    evals = []

    lissa = LiSSA(
        model,
        task,
        dataset['tr'],
        args['steps'],
        batch_size=args['batch_size'],
        damp=args['damp'],
        optimizer_options={'lr': args['lr']},
        lr_decay_power=0.0,
        use_tqdm=True
    )

    test_batch = dataset['_tokenizer'].batch_encode_plus(
        lst_fake[:1],
        padding=True,
        truncation=True,
        max_length=args['context'],  # Adjust this value based on your requirements
        return_tensors='pt'
    )
    test_batch = _transfer_batch_to_device(test_batch, torch.device(dist.get_rank()))

    dist.barrier()
    for text in lst_all[args['calc_from']: args['calc_to']]:
        model.zero_grad()
        # read prompt and continuation, and tokenize
        inputs = dataset['_tokenizer'](text, return_tensors="pt", max_length=args["context"], truncation=True)
        input_ids = inputs["input_ids"].to(dist.get_rank())
        attention_mask = inputs["attention_mask"].to(dist.get_rank())

        # Pass input_ids and attention_mask to the model
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        output.loss.backward()
        test_grad = iters.ContainerLike(iters.ParamIterator(model.parameters()))
        test_grad.copy_from(iters.GradIterator(model.parameters()))

        stest, curr_evals = lissa.run(
            test_grad,
            test_batch=test_batch,
            eval_every=args['eval_every']
        )

        evals.append(curr_evals)

        infs = []
        dot_infs = []

        for text in lst_all:
            model.zero_grad()
            # read prompt and continuation, and tokenize
            inputs = dataset['_tokenizer'](text, return_tensors="pt", max_length=args["context"], truncation=True)
            input_ids = inputs["input_ids"].to(dist.get_rank())
            attention_mask = inputs["attention_mask"].to(dist.get_rank())

            # Pass input_ids and attention_mask to the model
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            output.loss.backward()
            infs.append(stest.dot(iters.GradIterator(model.parameters())).item())
            dot_infs.append(test_grad.dot(iters.GradIterator(model.parameters())).item())

        influences.append(infs)
        dot_influences.append(dot_infs)

    if rank == 0:
        with open(output_filename, "w") as f:
            dct = {
                "args": args,
                "influences": influences,
                "dot_influences": dot_influences,
                "evals": evals
            }
            f.write(json.dumps(dct) + "\n")

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
        parser.add_argument(f"--calc_from", default=0, type=int)
        parser.add_argument(f"--calc_to", default=30, type=int)
        args = parser.parse_args()
        args = vars(args)
        args["log_dir"] = "logs/sim"

        # Create log dir if necessary and select log name
        os.makedirs(args["log_dir"], exist_ok=True)
        output_filename = os.path.join(args["log_dir"], f"seed_{args['seed']}_{args['exp_id']}.out")

        model_path = _NETWORK_PATHS[args['network']]['path']
        if "gpt2" in model_path:
            print("Loading real model")
            model_cpu = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            model_cpu = AutoModelForCausalLM.from_pretrained(model_path)

        model_cpu.eval()
        dataset = get_datasets(args)

        tp_utils.share_params_and_buffers(model_cpu)
        _args = (
            model_cpu,
            dataset,
            args,
            output_filename,
            n_gpus,
        )
        mp.spawn(run, args=_args, nprocs=n_gpus, join=True)
