from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from .sampler import SampleRateSampler, SampleRateBatchSampler


class SRL_BCELoss(nn.Module):
    def __init__(self, sampler: SampleRateSampler, optim='sgd', lr=0.1, momentum=0., weight_decay=0., norm=False, pos_rate=None, in_train=True, alternate=False):
        if not isinstance(sampler, SampleRateBatchSampler):
            raise TypeError

        super(SRL_BCELoss, self).__init__()
        self.sampler = sampler

        self.alpha = nn.Parameter(torch.tensor(0.).cuda())
        if pos_rate is None:
            self.pos_rate = self.alpha.sigmoid()
        else:
            self.pos_rate = pos_rate

        self.sampler.update(self.pos_rate)
        self.norm = norm
        self.in_train = in_train

        self.alternate = alternate

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

        self.train_losses = None
        self.val_losses = None

        self.initial = True

    def forward2(self, scores, labels: torch.Tensor):

        losses, is_pos = self.get_losses(scores, labels)

        pos_loss = losses[is_pos].mean()
        neg_loss = losses[~is_pos].mean()
        self.val_losses = [neg_loss, pos_loss]

        loss = losses.mean()

        # adjust pos_rate
        if isinstance(self.pos_rate, torch.Tensor):
            if self.initial:
                self.initial = False
                pos_rate = (pos_loss / (pos_loss + neg_loss)).detach()
                alpha = (pos_rate / (1. - pos_rate)).log().cpu().item()
                self.alpha.data = torch.tensor(alpha).cuda()
                # self.alpha = nn.Parameter(torch.tensor(alpha).cuda())
                # self.optimizer.param_groups[0]['params'] = [self.alpha]
            else:
                grad = (neg_loss - pos_loss).detach()
                if not torch.isnan(grad):
                    self.optimizer.zero_grad()
                    self.pos_rate.backward(grad)
                    self.optimizer.step()

            self.pos_rate = self.alpha.sigmoid()
            self.sampler.update(self.pos_rate)

        return loss

    def forward(self, scores, labels: torch.Tensor):
        if self.training and self.alternate:
            return self.forward2(scores, labels)

        losses, is_pos = self.get_losses(scores, labels)
        if is_pos.size(0) > self.sampler.batch_size:
            if not self.in_train:
                # use val data to estimate pos_loss and neg_loss
                val_losses = losses[self.sampler.batch_size:]
                val_is_pos = is_pos[self.sampler.batch_size:]
                train_is_pos = is_pos[:self.sampler.batch_size]
                pos_loss = val_losses[val_is_pos].mean()
                neg_loss = val_losses[~val_is_pos].mean()
                train_losses = losses[:self.sampler.batch_size]
                train_pos_loss = train_losses[train_is_pos].mean()
                train_neg_loss = train_losses[~train_is_pos].mean()
                self.train_losses = [train_neg_loss, train_pos_loss]
                self.val_losses = [neg_loss, pos_loss]
            else:
                val_losses = losses[self.sampler.batch_size:]
                val_is_pos = is_pos[self.sampler.batch_size:]
                val_pos_loss = val_losses[val_is_pos].mean()
                val_neg_loss = val_losses[~val_is_pos].mean()
                train_is_pos = is_pos[:self.sampler.batch_size]
                train_losses = losses[:self.sampler.batch_size]
                train_pos_loss = pos_loss = train_losses[train_is_pos].mean()
                train_neg_loss = neg_loss = train_losses[~train_is_pos].mean()
                self.train_losses = [neg_loss, pos_loss]
                self.val_losses = [val_neg_loss, val_pos_loss]

        else:
            train_pos_loss = pos_loss = losses[is_pos].mean()
            train_neg_loss = neg_loss = losses[~is_pos].mean()
            train_losses = losses
            self.train_losses = [neg_loss, pos_loss]
            # self.val_losses = None

        if self.norm:
            loss = (train_neg_loss + train_pos_loss) / 2.
        else:
            loss = train_losses.mean()

        # update pos_rate
        if self.training:
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