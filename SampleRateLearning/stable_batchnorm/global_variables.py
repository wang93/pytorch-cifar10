# encoding: utf-8
import torch
classes_num = 0
indices = [[]]
#braid_indices = [[]]
batch_size = 0
train_batch_size = 0


def parse_target(target):
    """each element of target ranges from 0. to (classes_num-1)"""
    global classes_num, indices, batch_size, train_batch_size  # , braid_indices

    if isinstance(target, list):
        pass
    elif isinstance(target, torch.Tensor):
        target = target.detach().view(-1).cpu().numpy().tolist()
    else:
        raise TypeError

    indices = [[] for _ in range(classes_num)]
    for i, e in enumerate(target[:train_batch_size]):
        indices[int(e)].append(i)
    # for i, e in enumerate(target[train_batch_size:]):
    #     indices[int(e)].append(i+train_batch_size)

    batch_size = len(target)
    # braid_indices = []
    # for sub_indices in indices:
    #     braid_sub_indices = sub_indices + [i + batch_size for i in sub_indices]
    #     braid_indices.append(braid_sub_indices)
    #
