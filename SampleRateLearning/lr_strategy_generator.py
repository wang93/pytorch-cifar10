
class PosRateLR(object):
    def __init__(self, optimizer):
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.optimizer = optimizer

    def step(self, pos_rate):
        pos_rate = pos_rate.min(1.-pos_rate) * 2.
        for p in self.optimizer.param_groups:
            p['lr'] = p['initial_lr'] * pos_rate
