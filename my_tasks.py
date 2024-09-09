import torch
import torch.nn.functional as F
import gnhtools

class ClassificationTask(gnhtools.BaseTask):
    """
        For classification problems. 
        Here batch = x, y, the latter is integer. 
        Output of the model has size [batch_size X n_classes]
    """
    def __init__(self, use_sampling_for_hessians=False):
        self.use_sampling_for_hessians = use_sampling_for_hessians
        self.dl_num_workers = 16

    def train_outputs(self, model, batch):
        x, y = batch['inputs'], batch['labels']
        return model(x)

    def train_loss_on_outputs(self, outputs, batch):
        x, y = batch['inputs'], batch['labels']
        return F.cross_entropy(outputs, y)

    def test_loss(self, model, batch, index=None):
        x, y = batch['inputs'], batch['labels']
        if index is not None:
            x, y = x[index:index+1, ...], y[index:index+1, ...]
        return F.cross_entropy(model(x), y) # TODO

    def batch_size(self, batch):
        return len(batch['labels'])
    
    def num_samples(self, batch):
        return self.batch_size(batch)

    def parse_batch(self, batch, index=None):
        if index is None:
            return [batch['inputs']], {}
        else:
            return [batch['inputs'][index:index+1, ...]], {}

    def get_collate_fn(self):
        return None

    def hessian_at_t(self, outputs, batch, t):
        softmax = F.softmax(outputs.detach(), dim=1)

        return t * softmax - softmax * torch.einsum('ij,ij->i', t, softmax).unsqueeze(1)

    def logits_from_outputs(self, outputs):
        return outputs


class LanguageModelTask(gnhtools.BaseTask):
    """
        For language modelling problems. 
        Here batch has fields 'input_ids', 'attention_mask'. 
        Output of the model has size [batch_size X max_length X num_tokens]
    """
    def __init__(self, use_sampling_for_hessians=False):
        self.use_sampling_for_hessians = use_sampling_for_hessians
        self.dl_num_workers = 0

    def train_outputs(self, model, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        return model.forward(input_ids, attention_mask=attention_mask)['logits']

    def logits_from_outputs(self, outputs):
        return outputs["logits"]

    def train_loss_on_outputs(self, outputs, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].float()
        loss = F.cross_entropy(outputs, input_ids, reduction='none')
        return (loss * attention_mask).sum() / attention_mask.sum()

    def test_loss(self, model, batch, index=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids
        if "labels" in batch:
            labels = batch['labels']
        if index is not None:
            input_ids = input_ids[index: index+1, ...]
            attention_mask = attention_mask[index: index+1, ...]
            labels = labels[index: index+1, ...]
        return model.forward(input_ids, attention_mask=attention_mask, labels=labels)['loss']

    def batch_size(self, batch):
        attention_mask = batch["attention_mask"]
        return attention_mask.sum().item()

    def num_samples(self, batch):
        return batch['input_ids'].size()[0]

    def parse_batch(self, batch, index=None):
        if index is None:
            return [batch['input_ids']], {"attention_mask": batch['attention_mask']}
        else:
            return [batch['input_ids'][index:index + 1, ...]], {"attention_mask": batch['attention_mask'][index:index + 1, ...]}

    def hessian_at_t(self, outputs, batch, t):
        attention_mask = batch['attention_mask']
        softmax = F.softmax(outputs.detach(), dim=2) * attention_mask.unsqueeze(2)

        return t * softmax - softmax * torch.einsum('ijk,ijk->ij', t, softmax).unsqueeze(2)
