# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/12/5 9:28

import torch
from torch.nn.modules.batchnorm import BatchNorm1d as origin_bn1d, BatchNorm2d as origin_bn2d

'''BN params learned in the first iteration, no affine'''


class BatchNorm1d(origin_bn1d):
    def __init__(self, eps=1e-5):
        super(origin_bn1d, self).__init__(128, eps, 1., False, True)

    def forward(self, input: torch.Tensor):
        if self.num_batches_tracked == 0:
            self.train()
            self.num_features = input.size(1)
            self.running_mean = torch.zeros(self.num_features).cuda()
            self.running_var = torch.zeros(self.num_features).cuda()
        else:
            self.eval()

        return super(BatchNorm1d, self).forward(input)


class BatchNorm2d(origin_bn2d):
    def __init__(self, eps=1e-5):
        super(origin_bn2d, self).__init__(128, eps, 1., False, True)

    def forward(self, input):
        if self.num_batches_tracked == 0:
            self.train()
            self.num_features = input.size(1)
            self.running_mean = torch.zeros(self.num_features).cuda()
            self.running_var = torch.zeros(self.num_features).cuda()
        else:
            self.eval()

        return super(BatchNorm2d, self).forward(input)


# def convert_model(module):
#     if isinstance(module, torch.nn.DataParallel):
#         mod = module.module
#         mod = convert_model(mod)
#         mod = torch.nn.DataParallel(mod, device_ids=module.device_ids)
#         return mod
#
#     mod = module
#     for pth_module, id_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
#                                       torch.nn.modules.batchnorm.BatchNorm2d],
#                                      [BatchNorm1d,
#                                       BatchNorm2d]):
#         if isinstance(module, pth_module):
#             mod = id_module(module.num_features, module.eps, 1., False, module.track_running_stats)
#             mod.running_mean = module.running_mean
#             mod.running_var = module.running_var
#             #if module.affine:
#             #    mod.weight.data = module.weight.data.clone().detach()
#             #    mod.bias.data = module.bias.data.clone().detach()
#
#     for name, child in module.named_children():
#         mod.add_module(name, convert_model(child))
#
#     return mod
