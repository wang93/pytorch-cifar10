from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW
from .sampler import SampleRateSampler, SampleRateBatchSampler


class SRI_BCELoss(nn.Module):
    def __init__(self, sampler: SampleRateSampler, norm=False):
        if not isinstance(sampler, SampleRateBatchSampler):
            raise TypeError

        super(SRI_BCELoss, self).__init__()

        self.pos_rate = 0.5
        self.sampler = sampler
        self.sampler.update(self.pos_rate)
        self.norm = norm
        self.recent_losses = None

    def forward(self, scores, labels: torch.Tensor):
        losses, is_pos = self.get_losses(scores, labels)
        pos_loss = losses[is_pos].mean()
        neg_loss = losses[~is_pos].mean()

        self.recent_losses = [pos_loss, neg_loss]

        if self.norm:
            if torch.isnan(pos_loss):
                print('pos_loss is nan!')
                loss = neg_loss * 0.
            elif torch.isnan(neg_loss):
                print('neg_loss is nan!')
                loss = pos_loss * 0.
            else:
                pos_num = is_pos.sum()
                batch_size = scores.size(0)
                real_pos_rate = pos_num / float(batch_size)
                scale_correction_factor = torch.sqrt(real_pos_rate * (1. - real_pos_rate))
                loss = (pos_loss + neg_loss) * scale_correction_factor

        else:
            loss = losses.mean()

        # inference pos_rate
        self.pos_rate = (pos_loss / (neg_loss + pos_loss + 0.000001)).cpu().item()

        return loss

    def get_losses(self, scores, labels: torch.Tensor):
        losses = nn.BCELoss(reduction='none')(scores.sigmoid(), labels)
        is_pos = labels.type(torch.bool)
        return losses, is_pos


class SRI_CELoss(SRI_BCELoss):
    def get_losses(self, scores, labels: torch.Tensor):
        labels = labels.to(dtype=torch.long).view(-1)
        losses = nn.CrossEntropyLoss(reduction='none')(scores, labels)
        is_pos = labels.type(torch.bool)
        return losses, is_pos


class SRL_BCELoss(nn.Module):
    def __init__(self, sampler: SampleRateSampler, optim='sgd', lr=0.1, momentum=0., weight_decay=0., norm=False, pos_rate=None):
        if not isinstance(sampler, SampleRateBatchSampler):
            raise TypeError

        super(SRL_BCELoss, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.).cuda())
        if pos_rate is None:
            self.pos_rate = self.alpha.sigmoid()
        else:
            self.pos_rate = pos_rate

        self.sampler = sampler
        self.sampler.update(self.pos_rate)
        self.norm = norm

        param_groups = [{'params': [self.alpha]}]
        if optim == "sgd":
            default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
            optimizer = SGD(param_groups, **default)

        elif optim == 'adam':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             amsgrad=False)

        elif optim == 'amsgrad':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             amsgrad=True)

        elif optim == 'adamw':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = AdamW(param_groups, **default,
                              betas=(0.9, 0.999),
                              eps=1e-8,
                              amsgrad=False)
        else:
            raise NotImplementedError

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.optimizer = optimizer

        self.train_losses = None
        self.val_losses = None

    def forward(self, scores, labels: torch.Tensor):
        losses, is_pos = self.get_losses(scores, labels)
        if is_pos.size(0) > self.sampler.batch_size:
            # use val data to estimate pos_loss and neg_loss
            val_losses = losses[self.sampler.batch_size:]
            val_is_pos = is_pos[self.sampler.batch_size:]
            train_is_pos = is_pos[:self.sampler.batch_size]
            pos_loss = val_losses[val_is_pos].mean()
            neg_loss = val_losses[~val_is_pos].mean()
            train_losses = losses[:self.sampler.batch_size]
            self.train_losses = [train_losses[~train_is_pos].mean(), train_losses[train_is_pos].mean()]
            self.val_losses = [neg_loss, pos_loss]

        else:
            pos_loss = losses[is_pos].mean()
            neg_loss = losses[~is_pos].mean()
            train_losses = losses
            self.train_losses = [neg_loss, pos_loss]
            self.val_losses = None

        if self.norm:
            raise NotImplementedError
        else:
            loss = train_losses.mean()

        # update pos_rate
        grad = (neg_loss - pos_loss).detach()
        if (not torch.isnan(grad)) and isinstance(self.pos_rate, torch.Tensor):
            self.optimizer.zero_grad()
            self.pos_rate.backward(grad)
            self.optimizer.step()
            self.pos_rate = self.alpha.sigmoid()

        self.sampler.update(self.pos_rate)

        return loss

    def get_losses(self, scores, labels: torch.Tensor):
        losses = nn.BCELoss(reduction='none')(scores.sigmoid(), labels)
        is_pos = labels.type(torch.bool)
        return losses, is_pos


class SRL_CELoss(SRL_BCELoss):
    def get_losses(self, scores, labels: torch.Tensor):
        labels = labels.to(dtype=torch.long).view(-1)
        losses = nn.CrossEntropyLoss(reduction='none')(scores, labels)
        is_pos = labels.type(torch.bool)
        return losses, is_pos