from torch import nn
import torch
from .loss import SRL_CELoss as ori_SRL_CELoss


class Equal_Gradient_SRL_CELoss(ori_SRL_CELoss):
    def __init__(self, *args, **kwargs):
        super(Equal_Gradient_SRL_CELoss, self).__init__(*args, **kwargs)

        if not self.alternate:
            raise NotImplementedError('only support alternate mode currently!')

        if self.norm:
            raise NotImplementedError('not support srl_norm currently!')

        self.softmax = torch.nn.Softmax(dim=1)

    def get_losses(self, scores, labels: torch.Tensor):
        raise NotImplementedError('this function is removed!')

    def forward2(self, scores, labels: torch.Tensor):
        labels = labels.to(dtype=torch.long).view(-1)
        is_pos = labels.type(torch.bool)
        losses = nn.CrossEntropyLoss(reduction='none')(scores, labels)

        pos_loss = losses[is_pos].mean()
        neg_loss = losses[~is_pos].mean()
        self.val_losses = [neg_loss, pos_loss]
        loss = losses.mean()
        # adjust pos_rate
        if isinstance(self.pos_rate, torch.Tensor):
            grad = (neg_loss - pos_loss).detach()
            if not torch.isnan(grad):
                self.optimizer.zero_grad()
                self.pos_rate.backward(grad)
                self.optimizer.step()

            self.pos_rate = self.alpha.sigmoid()
            self.sampler.update(self.pos_rate)
        return loss

    def forward(self, scores: torch.Tensor, labels: torch.Tensor):
        if self.training:
            return self.forward2(scores, labels)

        labels = labels.to(dtype=torch.long).view(-1)
        is_pos = labels.type(torch.bool)
        probs = self.softmax(scores.detach())
        g_pos = probs[is_pos, 0]
        g_neg = probs[~is_pos, 1]
        mean_g_pos = torch.mean(g_pos)
        mean_g_neg = torch.mean(g_neg)
        weight_base = torch.sum(g_pos) + torch.sum(g_neg) / float(self.sampler.batch_size)

        weight_pos = weight_base / mean_g_pos
        weight_neg = weight_base / mean_g_neg
        weights = torch.tensor([weight_neg, weight_pos]).cuda()
        losses = nn.CrossEntropyLoss(reduction='none', weight=weights)(scores, labels)

        pos_loss = losses[is_pos].mean()
        neg_loss = losses[~is_pos].mean()
        self.train_losses = [neg_loss, pos_loss]

        loss = losses.mean()

        return loss