from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from .sampler import SampleRateSampler, SampleRateBatchSampler


class SRL_BCELoss(nn.Module):
    def __init__(self, sampler: SampleRateSampler, optim='sgd', lr=0.1, momentum=0., weight_decay=0.,
                 pos_rate=None, in_train=True, alternate=False, soft_precision=False, alpha=None):
        if not isinstance(sampler, SampleRateBatchSampler):
            raise TypeError

        super(SRL_BCELoss, self).__init__()

        self.sampler = sampler

        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(0.).cuda())
        else:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)).cuda())

        self.in_train = in_train

        self.alternate = alternate
        self.soft_precision = soft_precision

        param_groups = [{'params': [self.alpha]}]
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
        if pos_rate is None:
            self.pos_rate = self.alpha.sigmoid()
            # self.pos_rate = self.alpha * 0.25
        else:
            self.pos_rate = pos_rate

        self.sampler.update(self.pos_rate)

        self.train_losses = None
        self.val_losses = None

        self.initial = True

    def __setattr__(self, key, value):
        super(SRL_BCELoss, self).__setattr__(key, value)

    def forward2(self, scores, labels: torch.Tensor):

        losses, is_pos = self.get_losses(scores, labels)

        pos_loss = losses[is_pos].mean()
        neg_loss = losses[~is_pos].mean()
        self.val_losses = [neg_loss, pos_loss]

        loss = losses.mean()

        # adjust pos_rate
        if isinstance(self.pos_rate, torch.Tensor):

            grad = (neg_loss - pos_loss).detach()
            if not torch.isnan(grad):
                self.pos_rate.backward(grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # self.optimizer.zero_grad()
                self.pos_rate = self.alpha.sigmoid()
                # self.pos_rate = self.alpha * 0.25
                self.sampler.update(self.pos_rate)

        return loss

    def forward(self, scores, labels: torch.Tensor):
        if self.training and self.alternate:
            return self.forward2(scores, labels)

        losses, is_pos = self.get_losses(scores, labels)
        if is_pos.size(0) > self.sampler.batch_size:
            raise NotImplementedError('This condition is abandoned!')

        train_pos_losses = losses[is_pos]
        train_neg_losses = losses[~is_pos]
        train_pos_loss = train_pos_losses.mean()
        train_neg_loss = train_neg_losses.mean()
        self.train_losses = [train_neg_loss, train_pos_loss]

        loss = losses.mean()

        return loss

    def get_losses(self, scores, labels: torch.Tensor):
        is_pos = labels.type(torch.bool)
        scores = scores.sigmoid()
        if self.soft_precision and self.alternate and self.training:
            # losses = scores
            losses = (scores > 0.5).to(dtype=torch.float)
            losses[is_pos] = 1. - losses[is_pos]
        else:
            losses = nn.BCELoss(reduction='none')(scores, labels)
        return losses, is_pos


class SRL_CELoss(SRL_BCELoss):
    def get_losses(self, scores, labels: torch.Tensor):
        labels = labels.to(dtype=torch.long).view(-1)
        is_pos = labels.type(torch.bool)
        if self.soft_precision and self.alternate and self.training:
            scores = torch.nn.functional.softmax(scores, dim=1)
            # losses = scores[:, 1].view(-1)
            losses = (scores[:, 1].view(-1) > 0.5).to(dtype=torch.float)
            losses[is_pos] = 1. - losses[is_pos]
        else:
            losses = nn.CrossEntropyLoss(reduction='none')(scores, labels)
        return losses, is_pos