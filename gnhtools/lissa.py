import torch
import gnhtools.iters as iters
import gnhtools.utils as utils
from gnhtools.hvp import hvp_at_batch

import tp_utils

from torch.optim.lr_scheduler import _LRScheduler

class PowerDecayLR(_LRScheduler):
    def __init__(self, optimizer, lr_0, power, last_epoch=-1):
        self.lr_0 = lr_0
        self.power = power
        super(PowerDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr_0 / ((self.last_epoch + 1) ** self.power) for _ in self.optimizer.param_groups]


class LiSSA:
    def __init__(self,
        model,
        task,
        dataset,
        steps,
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
        self.loader = tp_utils.BroadcastingDataLoader(
            dataset, 0,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=task.get_collate_fn(),
            num_workers=task.dl_num_workers,
            use_tqdm=use_tqdm
        )

        self.model = model
        self.task = task
        self.device_ = torch.device(f"cuda:{tp_utils.get_rank()}")

        self.optimizer_options = optimizer_options
        self.lr_decay_power = lr_decay_power

    def run(self,
        vec,
        test_batch=None,
        eval_every=0
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
            assert (
                test_batch is not None), "test_batch must be specified for eval_very != 0"
            evals = [[] for _ in range(self.task.num_samples(test_batch))]
        else:
            evals = None

        res = iters.ContainerLike(iters.ParamIterator(self.model))
        res.copy_from(vec) # starting from the vec correponds to gradient influence,
                           # so we can choose steps=0

        if abs(self.damp - 1.0) > 1e-6:
            res.scale(self.damp)
        res_params = list(map(torch.nn.Parameter, res))

        """
            res_params is a list of parameters. If we want to access either it's gradients or 
            data, we need to specify it with one of
            iters.GradIterator(res_params)
            iters.ParamIterator(res_params)
        """

        opt = torch.optim.SGD(res_params, weight_decay=0.0, **self.optimizer_options)
        scheduler = PowerDecayLR(opt, self.optimizer_options['lr'], self.lr_decay_power)

        for step, batch in enumerate(self.loader):
            memory_allocated_gb = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            self.loader.set_postfix(
                memory = "{:.1f}gb".format(memory_allocated_gb)
            )

            batch = utils._transfer_batch_to_device(batch, self.device_)
            hvp_at_batch(batch, self.model, self.task, iters.ParamIterator(res_params))

            for jvp_tensor, res_tensor, vec_tensor in zip(iters.GradIterator(self.model), res_params, vec):
                res_tensor.grad = self.invdamp * jvp_tensor + self.damp * res_tensor.data - vec_tensor
            opt.step()
            opt.zero_grad()
            scheduler.step()

            if (eval_every > 0) and (step % eval_every == 0):
                for i in range(self.task.num_samples(test_batch)):
                    self.model.zero_grad()
                    # read prompt and continuation, and tokenize

                    self.task.test_loss(self.model, test_batch, index=i).backward()
                    score = iters.GradIterator(self.model.parameters()).dot(res).item()
                    evals[i].append(score)

        return res, evals
