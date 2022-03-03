from __future__ import absolute_import
from torch.utils.data.sampler import Sampler
from queue import Queue
from random import sample as randsample
import torch
from random import shuffle
from copy import deepcopy
from numpy import clip


class _HalfQueue(object):
    def __init__(self, elements: list):
        self.recent = Queue(maxsize=len(elements))
        shuffle(elements)
        self.elements = elements
        self.queue = deepcopy(elements)
        #self.selection_pool = set(elements)

    # def _update(self, new_element):
    #     self.selection_pool.remove(new_element)
    #
    #     if self.recent.full() or len(self.selection_pool) == 0:
    #         old_element = self.recent.get()
    #         self.selection_pool.add(old_element)
    #
    #     self.recent.put(new_element)

    def select(self, num):
        res = []
        for i in range(num):
            if len(self.queue) > 0:
                e = self.queue.pop()
                res.append(e)
            else:
                self.queue = deepcopy(self.elements)
                shuffle(self.queue)
                e = self.queue.pop()
                res.append(e)

        return res


class SampleRateBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        super(SampleRateBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.sample_rates = None
        self.sample_num_per_epoch = len(data_source)
        self.batch_size = batch_size
        indices = [[] for _ in range(len(data_source.classes))]
        for i, t in enumerate(data_source.targets):
            indices[t].append(i)
        self.sample_agents = [_HalfQueue(sub_indices) for sub_indices in indices]
        self.length = (self.sample_num_per_epoch + self.batch_size - 1) // self.batch_size

        total_indices = []
        for idxs in indices:
            total_indices.extend(idxs)

        # self.instance_wise_sample_agent = _HalfQueue(total_indices, len(total_indices)-1)

    def update(self, sample_rates):
        if isinstance(sample_rates, list):
            self.sample_rates = sample_rates
        else:
            # self.sample_rates = sample_rates.detach().cpu().numpy().tolist()
            sample_rates = sample_rates.detach()
            # sample_rates = sample_rates / sum(sample_rates)
            self.sample_rates = sample_rates.cpu().numpy().tolist()

    def __iter__(self):
        self.cur_idx = -1
        return self

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx >= self.length:
            raise StopIteration

        # nums = [None for _ in self.sample_rates]
        # indices = torch.multinomial(torch.tensor(self.sample_rates), self.batch_size, replacement=True)
        # for i in range(len(self.sample_rates)):
        #     nums[i] = sum(indices == i).item()

        nums = [round(self.batch_size*r) for r in self.sample_rates]
        nums = [int(clip(n, 1, self.batch_size-1)) for n in nums]

        batch = []
        for agent, num in zip(self.sample_agents, nums):
            batch.extend(agent.select(num))

        return batch

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.length


class ValidationBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        super(ValidationBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        c2i = [[] for _ in range(len(data_source.classes))]
        for i, t in enumerate(data_source.targets):
            c2i[t].append(i)
        data_source.class_to_indices = c2i

        num_classes = len(c2i)
        if self.batch_size % num_classes != 0:
            raise ValueError
        self.num = self.batch_size // num_classes
        self.sample_agents = [_HalfQueue(sub_indices) for sub_indices in c2i]

    def __next__(self):
        batch = []
        for agent in self.sample_agents:
            batch += agent.select(self.num)

        return batch

    def __iter__(self):
        return self
