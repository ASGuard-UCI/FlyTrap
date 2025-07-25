from typing import Any, Sequence, Mapping

import numpy as np
import torch
from mmengine import FUNCTIONS

@FUNCTIONS.register_module()
def custom_collate_fn(data_batch: Sequence) -> Any:
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif data_item_type.__module__ == 'numpy' and data_item_type.__name__ != 'str_' \
            and data_item_type.__name__ != 'string_':
        # array of string classes and object
        # TODO: skip the condition, potentially leading to error
        # if data_item_type.__name__ == 'ndarray' \
        #         and np_str_obj_array_pattern.search(data_batch.dtype.str) is not None:
        #     return data_batch
        data_batch = np.stack(data_batch) # speed up
        return torch.as_tensor(data_batch)
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(custom_collate_fn(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [custom_collate_fn(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [custom_collate_fn(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [custom_collate_fn(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: custom_collate_fn([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return data_batch

