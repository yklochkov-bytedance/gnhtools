import torch
import gnhtools.utils as utils
from gnhtools.lissa import PowerDecayLR

import tp_utils
from torch.distributed._tensor import DTensor

class PBRF:
    def __init__(self,
        model,
        model_copy,
        task,
        dataset,
        steps,
        device_mesh,
        batch_size=16,
        damp=0.01,
        optimizer_options={},
        lr_decay_power=0.0,
        use_tqdm=True):

        self.damp, self.invdamp = damp, 1.0
        if damp > 1.0:
            self.invdamp, self.damp = 1.0 / damp, 1.0 # TODO: note that it's fixed!

        self.steps = steps
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=steps * batch_size)
        self.loader = tp_utils.BroadcastingDataLoader(dataset, 0, batch_size=batch_size, sampler=sampler, collate_fn=task.get_collate_fn(), num_workers=task.dl_num_workers, use_tqdm=use_tqdm)

        self.model = model
        self.model_copy = model_copy
        self.task = task
        self.device_ = torch.device(f"cuda:{tp_utils.get_rank()}")
        self.device_mesh = device_mesh

        self.optimizer_options = optimizer_options
        self.lr_decay_power = lr_decay_power

    def get_divergence(self, model, model_copy, batch):
        with torch.no_grad():
            logits_orig = self.task.train_outputs(model, batch).detach()

        logits = self.task.train_outputs(model_copy, batch)
        res = (torch.log(torch.exp(logits).sum(dim=-1)) - torch.log(torch.exp(logits_orig).sum(dim=-1))).sum() \
            - torch.sum(torch.nn.functional.softmax(logits_orig, dim=-1) * (logits - logits_orig)) 
        return res / self.task.batch_size(batch)

    def get_proximity_penalty(self, model, model_copy):
        # already halfed
        res = 0.0
        for par, par_orig in zip(model.parameters(), model_copy.parameters()):
            if not par.requires_grad:
                continue
            res -= torch.tensordot(par, par_orig.detach(), dims=len(par.size()))
            res += 0.5 * torch.tensordot(par, par, dims=len(par.size()))
        return res
    
    def get_point_loss(self, model_copy, batch):
        return self.task.test_loss(model_copy, batch)

    def run(self,
        train_batch,
        test_batch=None,
        epsilon=0.01,
        eval_every=0,
        ):
        """
            model:
            vec: TensorIterator, must not be derived from model.grad
            dataset: training dataset, on which we calculate Hessian/ Gauss-Newton Hessian
            steps: number of steps
            gnh: if true, use Gauss-Newton Hessian approximation, else use exact Hessian
            batch_size: batch size for evaluation
            eval_every: evaluate loss every eval_every steps
            optimizer: adam or sgd
            optimizer_options: dictionary of options for the correponding optimizer
        """
        if eval_every > 0:
            assert (test_batch is not None
            ), "test_batch must be specified for eval_very != 0"
            inputs, optional_inputs = self.task.parse_batch(test_batch)
            n_test = inputs[0].size()[0]
            evals = [[] for _ in range(n_test)]
        else:
            evals = None

        for par, par_copy in zip(self.model.parameters(), self.model_copy.parameters()):
            if isinstance(par.data, DTensor):
                par_copy.data._local_tensor[...] = par.data._local_tensor
            else:
                par_copy.data[...] = par.data

        opt = torch.optim.SGD([p for p in self.model_copy.parameters() if p.requires_grad], weight_decay=0.0, **self.optimizer_options)
        scheduler = PowerDecayLR(opt, self.optimizer_options['lr'], self.lr_decay_power)

        train_batch = utils._transfer_batch_to_device(train_batch, self.device_)
        test_batch = utils._transfer_batch_to_device(test_batch, self.device_)

        outputs_tests = []
        for i in range(n_test):
            with torch.no_grad():
                outputs_tests.append(self.task.test_loss(
                    self.model, test_batch, index=i).detach())

        for step, batch in enumerate(self.loader):
            memory_allocated_gb = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            self.loader.set_postfix(
                memory = "{:.1f}gb".format(memory_allocated_gb)
            )

            batch = utils._transfer_batch_to_device(batch, self.device_)
            div = self.get_divergence(self.model, self.model_copy, batch)
            point = self.get_point_loss(self.model_copy, train_batch)
            #prox = self.get_proximity_penalty(self.model, self.model_copy)

            (self.invdamp * div + epsilon * point).backward()

            # add proximity gradient
            for par, par_copy in zip(self.model.parameters(), self.model_copy.parameters()):
                if not par.requires_grad:
                    continue
                if isinstance(par.data, DTensor):
                    par_copy.grad._local_tensor[...] += self.damp * (
                        par_copy.data._local_tensor - par.data._local_tensor)
                else:
                    par_copy.grad[...] += self.damp * (par_copy.data - par.data)

            opt.step()
            opt.zero_grad()
            scheduler.step()

            if ((eval_every > 0) and (step % eval_every == 0 or 
                step == self.steps - 1)
                ):

                # Pass input_ids and attention_mask to the model
                for i in range(n_test):
                    with torch.no_grad():
                        outputs_test_new = self.task.test_loss(self.model_copy, test_batch, index=i).detach()
                    evals[i].append((outputs_test_new - outputs_tests[i]).item() / epsilon)

        return evals
