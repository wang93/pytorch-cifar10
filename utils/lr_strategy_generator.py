
class MileStoneLR_WarmUp(object):
    def __init__(self, optimizer, milestones=None, gamma=0.5, warmup_till=1, warmup_mode='linear'):
        if milestones is None:
            milestones = [75, 150]

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_till = warmup_till
        self.warmup_mode = warmup_mode

    def step(self, ep_from_1):
        if ep_from_1 < self.warmup_till:
            if self.warmup_mode == 'linear':
                mul = float(ep_from_1) / float(self.warmup_till)
            elif self.warmup_mode == 'const':
                mul = 0.1
            else:
                raise NotImplementedError

        else:
            i = 0
            for i, m in enumerate(self.milestones):
                if ep_from_1 <= m:
                    break

            if ep_from_1 > self.milestones[-1]:
                i += 1

            mul = self.gamma ** i

        for p in self.optimizer.param_groups:
            p['lr'] = p['initial_lr'] * mul
