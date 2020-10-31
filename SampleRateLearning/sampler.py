from __future__ import absolute_import

from collections import defaultdict

from numpy import clip
from torch.utils.data.sampler import Sampler

from queue import Queue
from random import sample as randsample


class SampleRateSampler(Sampler):
    def __init__(self, data_source):
        super(SampleRateSampler, self).__init__(data_source)
        self.data_source = data_source
        # self.alpha = torch.nn.Parameter(torch.tensor(0.))
        self.pos_rate = 0.5
        self.sample_num_per_epoch = len(data_source)

    def update(self, pos_rate):
        self.pos_rate = pos_rate.cpu().item()

    def __iter__(self):
        self.cur_idx = -1
        return self

    def __next__(self):
        raise NotImplementedError

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.sample_num_per_epoch


class _HalfQueue(object):
    def __init__(self, elements: list):
        num_elements = len(elements)
        max_recent_num = num_elements // 2
        self.recent = Queue(maxsize=max_recent_num)
        self.selection_pool = set(elements)

    def _update(self, new_element):
        self.selection_pool.remove(new_element)

        if self.recent.full():
            old_element = self.recent.get()
            self.selection_pool.add(old_element)

        self.recent.put(new_element)

    def select(self, num):
        res = randsample(self.selection_pool, num)
        for e in res:
            self._update(e)

        return res


class SampleRateBatchSampler(SampleRateSampler):
    def __init__(self, data_source, batch_size=1):
        super(SampleRateBatchSampler, self).__init__(data_source)
        self.batch_size = batch_size
        indices = [[], []]
        for i, t in enumerate(data_source.targets):
            indices[t].append(i)
        self.sample_agents = [_HalfQueue(sub_indices) for sub_indices in indices]
        self.length = (self.sample_num_per_epoch + self.batch_size - 1) // self.batch_size

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx >= self.length:
            raise StopIteration

        a_num = round(self.batch_size * self.pos_rate)
        a_num = int(clip(a_num, 1, self.batch_size-1))
        b_num = self.batch_size - a_num
        batch = self.sample_agents[0].select(a_num) + self.sample_agents[1].select(b_num)

        return batch

    def __len__(self):
        return self.length