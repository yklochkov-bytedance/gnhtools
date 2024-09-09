import torch

def jvp_and_hjvp_at_batch(batch, model, task, vector, DELTA=0.01, n_chunks=1):
    """
        Computes add-hoc values.
        TODO: add formula what it computes

        Arguments:
            batch: dict, tuple, etc.,
            model: nn.Module,
            task: BaseTask,
            vector: tuple or list of Tensors,

        Result:
            return none, results will be saved in model's parameters' gradients, i.e. par.grad for par in model.parameters()
    """

    inputs, optional_inputs = task.parse_batch(batch)

    with torch.no_grad():
        outputs = model(*inputs, **optional_inputs)
        logits = task.logits_from_outputs(outputs).detach()

    def perturebed_params(indices, delta):
        res = {}
        for name, buff in model.named_buffers():
                res[name] = buff
        vector_iter = iter(vector)
        i = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                res[name] = param.data
            else:
                v = next(vector_iter)
                if i in indices:
                    res[name] = param.data + v * delta
                else:
                    res[name] = param.data
                i += 1
        return res

    def get_jvp_at_indices(indices):
        with torch.no_grad():
            perturbed_ = perturebed_params(indices, DELTA)

            outputs_perturbed = torch.func.functional_call(model, perturbed_, *inputs, kwargs=optional_inputs)
            logits_perturbed = task.logits_from_outputs(outputs_perturbed).detach()

            perturbed_ = perturebed_params(indices, -DELTA)

            outputs_perturbed_ = torch.func.functional_call(model, perturbed_, *inputs, kwargs=optional_inputs)
            logits_perturbed_ = task.logits_from_outputs(outputs_perturbed_).detach()

            jvp_values = (logits_perturbed - logits_perturbed_) /  (2 * DELTA)

        return jvp_values

    n_params = sum([1 for p in model.parameters() if p.requires_grad])
    jvp_values = torch.zeros_like(logits)
    
    for chunk in range(n_chunks):
        per_chunk = (n_params + n_chunks - 1) // n_chunks
        idx = list(range(chunk * per_chunk, (chunk + 1) * per_chunk))
        jvp_values += get_jvp_at_indices(idx)

    hess_dot = task.hessian_at_t(logits, batch, jvp_values)
    return jvp_values, hess_dot, logits

def hvp_at_batch(batch, model, task, vector, do_zero_grad=True, scale=1.0):
    """
        Computes Gauss-Newton Hessian vector product on a batch.

        Arguments:
            batch: dict, tuple, etc.,
            model_func: FuncModelWrapper,
            task: BaseTask,
            vector: tuple or list of Tensors,
            do_zero_grad: bool, if True, zero gradients before backward pass.
            scale: float, scale factor for the vector

            J S (J^{T} v)  J = [ \ nabla f_j()^T v ]

        Result:
            return jvp_values, hess_dot; the actual result of the HVP will be recorded in model's parameters' gradients,
            i.e. par.grad for par in model.parameters(); use option do_zero_grad=False, in case you want to accumulate
        
        The result
    """

    # get jvp values
    jvp_values, hess_dot, _ = jvp_and_hjvp_at_batch(batch, model, task, vector) # jvp = [\nabla_t f_{ij}(t)^T vec]

    if do_zero_grad:
        model.zero_grad()
    inputs, optional_inputs = task.parse_batch(batch)
    logits = task.logits_from_outputs(model(*inputs, **optional_inputs))
    batch_size = task.batch_size(batch)
    fake_loss = torch.sum(logits * hess_dot) / batch_size
    fake_loss.backward()

    return jvp_values, hess_dot
