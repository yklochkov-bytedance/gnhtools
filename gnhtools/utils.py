import torch

def _transfer_batch_to_device(batch, device):
    """
        transfer a tuple, list, or dict of tensors to device
        device can be either int (denoting index of GPU) or torch.device instance
    """
    if isinstance(device, int):
        device = torch.device(device)
    if isinstance(batch, tuple):
        return tuple(_transfer_batch_to_device(item, device) for item in batch)
    elif isinstance(batch, list):
        return [_transfer_batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {key: _transfer_batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        try:
            return batch.to(device)
        except:
            raise TypeError(
            f"Cannot send batch to device, got {type(batch)}."
            f" Expect one of: `tuple`, `list`, `torch.Tensor`."
            f" Otherwise, implement batch.to(device) for your custom type."
        )
