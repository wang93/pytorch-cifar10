# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/16 15:30

"""
SummaryWriters
"""

from torch.utils.tensorboard import SummaryWriter
import torch
from os.path import join as pjoin
from SampleRateLearning.loss import SRL_CELoss


def normal_scalar(v):
    if isinstance(v, torch.Tensor):
        v = v.cpu().item()
    return v


class SummaryWriters(object):
    def __init__(self, opt, class_num):
        exp_dir = pjoin('./exps', opt.exp, 'tensorboard_log')
        self.summary_writer = SummaryWriter(pjoin(exp_dir, 'common'))
        self.max_summary_writer = SummaryWriter(pjoin(exp_dir, 'max'))
        self.min_summary_writer = SummaryWriter(pjoin(exp_dir, 'min'))
        self.avg_summary_writer = SummaryWriter(pjoin(exp_dir, 'avg'))

        self.class_summary_writers = []

        for i in range(class_num):
            self.class_summary_writers.append(SummaryWriter(pjoin(exp_dir, str(i))))

    def record_epoch(self, acc, precisions, global_step):
        self.summary_writer.add_scalar('accuracy', acc, global_step)
        self.summary_writer.add_scalar('worst precision', normal_scalar(min(precisions)), global_step)
        for writer, precision in zip(self.class_summary_writers, precisions):
            writer.add_scalar('test_precisions', normal_scalar(precision), global_step)

    def record_iter(self, loss, global_step, optimizer=None, criterion=None):
        if loss is not None:
            self.summary_writer.add_scalar('loss', normal_scalar(loss), global_step)

        if isinstance(criterion, SRL_CELoss):
            if criterion.train_losses is not None:
                for writer, t_loss in zip(self.class_summary_writers, criterion.train_losses):
                    writer.add_scalar('train_losses', normal_scalar(t_loss), global_step)

            if criterion.sample_rates is not None:
                for writer, sample_rate in zip(self.class_summary_writers, criterion.sample_rates):
                    writer.add_scalar('sample_rate', normal_scalar(sample_rate), global_step)

            if criterion.alphas is not None:
                for writer, alpha in zip(self.class_summary_writers, criterion.alphas):
                    writer.add_scalar('alpha', normal_scalar(alpha), global_step)

            if criterion.val_losses is not None:
                for writer, v_loss in zip(self.class_summary_writers, criterion.val_losses):
                    writer.add_scalar('val_losses', normal_scalar(v_loss), global_step)

        if optimizer is not None:
            cur_lr = optimizer.param_groups[0]['lr']
            self.summary_writer.add_scalar('lr', normal_scalar(cur_lr), global_step)
