import torch

import datasets
from transformers import AutoTokenizer, AutoModel

import argparse
import os

from networks import _NETWORK_PATHS
from dataset import _DATASET_PATHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="facebook/opt-1.3b",
                        help="Please provide network name for loading HF tokenizer")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting which index? Default value 0, i.e. from start.")
    parser.add_argument("--end_idx", type=int, default=50000,
                        help="Index of the end entry. Default value 10000 just for testing purposes.")
    args = parser.parse_args()

    tokenized_dataset_name = {
        "facebook/opt-1.3b": "web-text-tok-opt",
        "mistralai/Mistral-7B-v0.1": "web-text-tok-mistral",
    }[args.network]

    tokenized_dataset = datasets.load_from_disk(_DATASET_PATHS[tokenized_dataset_name]["path"])
    print(f"Total size of the dataset: {len(tokenized_dataset)}. Selecting range {args.start_idx} to {args.end_idx}")
    tokenized_dataset = tokenized_dataset.select(range(args.start_idx, args.end_idx))
    tokenizer_network = AutoTokenizer.from_pretrained(_NETWORK_PATHS[args.network]["path"])

    embedding_hf_name = "distilbert/distilbert-base-uncased"
    embedding_save_name = f"{os.environ['MOUNT_PATH']}/hugginface/models/distilbert"
    try:
        tokenizer_embedding = AutoTokenizer.from_pretrained(embedding_save_name)
        model_embedding = AutoModel.from_pretrained(embedding_save_name)
    except:
        print("Downloading DistilBERT...")
        tokenizer_embedding = AutoTokenizer.from_pretrained(embedding_hf_name)
        model_embedding = AutoModel.from_pretrained(embedding_hf_name)
        tokenizer_embedding.save_pretrained(embedding_save_name)
        model_embedding.save_pretrained(embedding_save_name)

    model_embedding = model_embedding.half().to(0)
    #model_embedding = torch.compile(model_embedding)
    

    def split_sum(mat):
        d = mat.shape[-1] // 4
        return (mat[..., :d] + mat[..., d:2*d] + mat[..., 2*d:3*d] + mat[..., 3*d: ]) / 4

    def get_text(batch):
        texts = tokenizer_network.batch_decode(batch['input_ids'])
        return {"text": texts}

    def get_embedding(batch):
        inputs = tokenizer_embedding.batch_encode_plus(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs.to(0)
        with torch.no_grad():
            outputs = model_embedding(**inputs).last_hidden_state.mean(axis=1).detach().cpu().numpy()
            outputs = split_sum(outputs)
        return {"embeddings": outputs}

    tokenized_dataset = tokenized_dataset.map(get_text, batched=True, batch_size=512, num_proc=16)
    tokenized_dataset = tokenized_dataset.map(get_embedding, batched=True, batch_size=512, num_proc=None)
    tokenized_dataset = tokenized_dataset.remove_columns(['input_ids', 'attention_mask', 'overflow_to_sample_mapping', "text"])
    print(f"cuda.max_memory_allocated = {torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024 : .1f}gb")

    print(tokenized_dataset)
    save_path = _DATASET_PATHS[tokenized_dataset_name]["path"]
    # remove ".hf" and add "_emb.hf"
    save_path = save_path[:-3] + f"-emb-{args.start_idx}-{args.end_idx}.hf"
    print(save_path)
    tokenized_dataset.save_to_disk(save_path)
