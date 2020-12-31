
from torch.autograd import Function
from torch.nn import Module
import torch
from .global_variables import indices, batch_size


class grad_scaling(Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x = args[0]
        scaling_factors = args[1]
        ctx.save_for_backward(scaling_factors)
        return x

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_x = grad_outputs[0]
        scaling_factors = ctx.saved_tensors[0]
        grad_x = grad_x * scaling_factors

        return grad_x, None


class Grad_Scaling(Module):
    def __init__(self, nums):
        super(Grad_Scaling, self).__init__()
        amount = sum(nums)
        self.target_ratios = [float(num)/float(amount) for num in nums]

    def forward(self, input):
        scaling_factors = torch.zeros(batch_size).to(device=input.device)
        for cls_indices, target_ratio in zip(indices, self.target_ratios):
            cur_ratio = float(len(cls_indices)) / float(batch_size)
            scaling_factor = target_ratio / cur_ratio
            scaling_factors[cls_indices] = scaling_factor

        if len(input.shape) == 4:
            scaling_factors = scaling_factors.view(-1, 1, 1, 1)
        elif len(input.shape) == 2:
            scaling_factors = scaling_factors.view(-1, 1)
        else:
            raise NotImplementedError

        return grad_scaling()(input, scaling_factors)




