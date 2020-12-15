import torch
from torch.nn.modules.batchnorm import BatchNorm1d as origin_bn1d, BatchNorm2d as origin_bn2d

'''Decreasing momentum, no affine'''


class BatchNorm1d(origin_bn1d):
    def __init__(self, eps=1e-5, base_momentum=1e-5):
        super(BatchNorm1d, self).__init__(128, eps, 1., False, True)
        self.base_momentum = base_momentum

    def forward(self, input: torch.Tensor):
        if self.training:
            self.momentum = self.base_momentum + (1. - self.base_momentum) ** (self.num_batches_tracked+1)

        return super(BatchNorm1d, self).forward(input)


class BatchNorm2d(origin_bn2d):
    def __init__(self, eps=1e-5, base_momentum=0.01):
        super(BatchNorm2d, self).__init__(128, eps, 1., False, True)
        self.base_momentum = base_momentum

    def forward(self, input: torch.Tensor):
        if self.training:
            self.momentum = self.base_momentum + (1. - self.base_momentum) ** (self.num_batches_tracked + 1)

        return super(BatchNorm2d, self).forward(input)