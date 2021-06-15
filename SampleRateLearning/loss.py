from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from .sampler import SampleRateBatchSampler


def get_rates(alphas):
    # T = 1.
    # return (alphas / T).softmax(dim=0)
    return alphas.sigmoid()

    # intensities = nn.ELU()(alphas) + 1.
    # rates = intensities / intensities.sum()
    # return rates


class SRL_CELoss(nn.Module):
    def __init__(self, sampler: SampleRateBatchSampler, optim='sgd', lr=0.1, momentum=0., weight_decay=0.,
                 sample_rates=None, precision_super=False):
        if not isinstance(sampler, SampleRateBatchSampler):
            raise TypeError

        super(SRL_CELoss, self).__init__()

        self.sampler = sampler

        self.num_classes = len(sampler.sample_agents)

        self.alphas = nn.Parameter(torch.zeros(self.num_classes).cuda())

        self.precision_super = precision_super

        param_groups = [{'params': [self.alphas]}]
        if optim == "sgd":
            default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
            optimizer = SGD(param_groups, **default)

        elif optim == 'adam':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0., 0.999),
                             eps=1e-8,
                             amsgrad=False)

        elif optim == 'sadam':
            from utils.optimizers import SAdam
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = SAdam(param_groups, **default,
                              betas=(0., 0.999),
                              eps=1e-8,
                              amsgrad=False)

        elif optim == 'amsgrad':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0., 0.999),
                             eps=1e-8,
                             amsgrad=True)

        elif optim == 'adamw':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = AdamW(param_groups, **default,
                              betas=(0., 0.999),
                              eps=1e-8,
                              amsgrad=False)

        elif optim == 'rmsprop':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = RMSprop(param_groups, **default,
                                alpha=0.999,
                                eps=1e-8)

        elif optim == 'rmsprop2':
            from utils.optimizers import RMSprop2
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = RMSprop2(param_groups, **default,
                                 alpha=0.999,
                                 eps=1e-8)

        elif optim == 'rmsprop3':
            from utils.optimizers import RMSprop3
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = RMSprop3(param_groups, **default,
                                 alpha=0.999,
                                 eps=1e-8)

        elif optim == 'adammw':
            from utils.optimizers import AdamMW
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = AdamMW(param_groups, **default,
                               betas=(0., 0.999),
                               eps=1e-8,
                               amsgrad=False)
        else:
            raise NotImplementedError

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.optimizer = optimizer

        self.optimizer.zero_grad(set_to_none=True)
        # self.optimizer.zero_grad()
        if sample_rates is None:
            # self.sample_rates = self.alphas.softmax(dim=0)
            self.sample_rates = get_rates(self.alphas)
        else:
            self.sample_rates = sample_rates

        self.sampler.update(self.sample_rates)

        self.train_losses = None
        self.val_losses = None

    def forward2(self, scores, labels: torch.Tensor):
        # losses, labels = self.get_losses(scores, labels)
        labels = labels.to(dtype=torch.long).view(-1)
        if self.precision_super:
            scores = scores.softmax(dim=1)
            # scores = scores[list(range(scores.size(0))), labels]
            # losses = (scores < 0.5).to(dtype=torch.float)
            predictions = torch.argmax(scores, dim=1, keepdim=False)
            losses = (predictions != labels).to(dtype=torch.float)
        else:
            raise NotImplementedError
            losses = nn.CrossEntropyLoss(reduction='none')(scores, labels)

        self.val_losses = []
        for i in range(self.num_classes):
            prediction_mask = (predictions == i)
            label_mask = (labels == i)
            intersection = torch.bitwise_and(prediction_mask, label_mask).to(dtype=torch.float).sum()
            union = torch.bitwise_or(prediction_mask, label_mask).to(dtype=torch.float).sum()

            if union == 0:
                iou = 0.
            else:
                iou = intersection / union

            self.val_losses.append(iou)

            # cur_mask = (predictions == i)
            # cur_precisions = 1. - losses[cur_mask]
            # cur_mask = (labels == i)
            # cur_recalls = 1. - losses[cur_mask]
            # if len(cur_precisions) == 0:
            #     print('precision of the {0}th class can not be computed!'.format(i))
            #     cur_precision = torch.tensor(0.5).cuda()
            # else:
            #     cur_precision = cur_precisions.mean()
            # if len(cur_recalls) == 0:
            #     print('hit2')
            #     cur_recall = torch.tensor(0.5).cuda()
            # else:
            #     cur_recall = cur_recalls.mean()
            #
            # cur_f1 = 2*cur_precision*cur_recall/(cur_precision+cur_recall+0.0001)

            # cur_f1 = (cur_precision * cur_recall).sqrt()
            # print(cur_precision)
            # print(cur_recall)
            # print('---------------')

            # self.val_losses.append(1.-cur_f1)
        self.val_losses = torch.Tensor(self.val_losses).cuda()

        loss = losses.mean()
        if torch.isnan(loss):
            raise ValueError

        # adjust sample_rates
        if isinstance(self.sample_rates, torch.Tensor):
            grad = self.val_losses.detach().mean() - self.val_losses.detach()
            self.sample_rates.backward(grad)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # self.sample_rates = self.alphas.softmax(dim=0)
            self.sample_rates = get_rates(self.alphas)
            self.sampler.update(self.sample_rates)

        return loss

    def forward(self, scores, labels: torch.Tensor):

        if self.training:
            return self.forward2(scores, labels)

        # losses, labels = self.get_losses(scores, labels)
        labels = labels.to(dtype=torch.long).view(-1)
        losses = nn.CrossEntropyLoss(reduction='none')(scores, labels)

        self.train_losses = []
        for i in range(self.num_classes):
            cur_mask = (labels == i)
            cur_losses = losses[cur_mask]
            cur_loss = cur_losses.mean()
            self.train_losses.append(cur_loss)

        loss = losses.mean()

        return loss