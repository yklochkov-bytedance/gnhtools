import torch

import gnhtools.iters as iters
import gnhtools.utils as utils
from gnhtools.hvp import hvp_at_batch, jvp_and_hjvp_at_batch

import tp_utils

from tqdm import tqdm

class GNHStats:
    def __init__(self, dataset, task, batch_size, n_samples, dl_num_workers=0):
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=2 * batch_size * n_samples)
        self.loader = tp_utils.BroadcastingDataLoader(
            dataset, 0,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=task.get_collate_fn(),
            num_workers=dl_num_workers,
            use_tqdm=True
        )

        self.rank = tp_utils.get_rank()

        # set consecutive seeds TODO: fix to be an option
        self.seeds = [(n_samples * self.rank + i) for i in range(n_samples)]
        self.task = task

    def get_first_order_stats(self, model):
        res = []
        for i, batch in enumerate(self.loader):
            batch = utils._transfer_batch_to_device(batch, self.rank)
            vec = iters.RandomLike(iters.ParamIterator(model.parameters()), self.seeds[i // 2])
            jvp_values, hess_dot, _ = jvp_and_hjvp_at_batch(batch, model, self.task, vec)
            res.append(torch.sum(jvp_values * hess_dot).item() / self.task.batch_size(batch))

        return res

    def get_second_order_stats(self, model):
        left = iters.ContainerLike(iters.ParamIterator(model.parameters()))
        res = []
        res_ = []
        res_gap = []
        for i, batch in enumerate(self.loader):
            batch = utils._transfer_batch_to_device(batch, self.rank)
            vec = iters.RandomLike(iters.ParamIterator(model.parameters()), self.seeds[i // 2])
            hvp_at_batch(batch, model, self.task, vec)
            if i % 2 == 0:
                left.copy_from(iters.GradIterator(model.parameters()))
                res_.append(left.dot(vec).item())
            else:
                res.append(left.dot(iters.GradIterator(model.parameters())).item())
                gap_val = left.norm() ** 2 + iters.GradIterator(model.parameters()).norm() ** 2
                gap_val = gap_val.item() / 2 - res[-1]
                res_gap.append(gap_val)

        return res, res_, res_gap

    def _stats_in_direction(self, model, vec):
        res = []
        for batch in self.loader:
            batch = utils._transfer_batch_to_device(batch, self.rank)
            jvp_values, hess_dot, _ = jvp_and_hjvp_at_batch(batch, model, self.task, vec)
            res.append(torch.sum(jvp_values * hess_dot).item() / self.task.batch_size(batch))

        return res


class GNHSketch:
    def __init__(self, dataset, task, batch_size, n_samples, n_columns):
        self.n_samples = n_samples
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=batch_size * n_samples * n_columns)
        self.loader = tp_utils.BroadcastingDataLoader(
            dataset, 0,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=task.get_collate_fn(),
            num_workers=task.dl_num_workers,
            use_tqdm=False)

        self.loader = iter(self.loader)

        self.rank = tp_utils.get_rank()
        self.task = task

    def _calc_column(self, model, projector_left, column):
        if self.n_samples > 1:
            res = iters.ContainerLike(column)
        else:
            res = None
        for _ in range(self.n_samples):
            batch = next(self.loader)
            batch = utils._transfer_batch_to_device(batch, self.rank)
            _, _ = hvp_at_batch(batch, model, self.task, column)
            if self.n_samples > 1:
                res.add_(iters.GradIterator(model), scale=1/self.n_samples)
            else:
                res = iters.GradIterator(model)
        if projector_left is not None:
            return projector_left.project(res)
        else:
            return res

    def _calc_column_sample_squared(self, model, projector_left, column):
        if self.n_samples > 1:
            res = iters.ContainerLike(column)
        else:
            res = None
        for batch in self.loader:
            batch = utils._transfer_batch_to_device(batch, self.rank)
            _, _ = hvp_at_batch(batch, model, self.task, column)
            step_one = list(iters.GradIterator(model))
            _, _ = hvp_at_batch(batch, model, self.task, step_one)

            if self.n_samples > 1:
                res.add_(iters.GradIterator(model), scale=1/self.n_samples)
            else:
                res = iters.GradIterator(model)
        return projector_left.project(res)

    def calc_hessian(self, model, projector_left, projector_right, columns_from=None, columns_to=None, sample_squared=False, squared=False):
        if columns_to is None:
            columns_from, columns_to = 0, projector_right.dim

        column = iters.ContainerLike(projector_right.get_column(0))
        res = torch.zeros(projector_left.dim, columns_to - columns_from, device=self.rank)
        range_from_to = range(columns_from, columns_to)
        if self.rank == 0:
            range_from_to = tqdm(range_from_to)
        for i, column_no in enumerate(range_from_to):
            memory_allocated_gb = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            if self.rank == 0:
                range_from_to.set_postfix(
                    memory = "{:.1f}gb".format(memory_allocated_gb)
                )
            column.copy_from(projector_right.get_column(column_no))
            if sample_squared:
                res[:, i] = self._calc_column_sample_squared(
                    model, projector_left, projector_right.get_column(column_no))
            elif squared:
                hvp = self._calc_column(
                    model, None, projector_right.get_column(column_no))
                res[:, i] = self._calc_column(
                    model, projector_left, hvp)
            else:
                res[:, i] = self._calc_column(
                    model, projector_left, projector_right.get_column(column_no))
        return res
