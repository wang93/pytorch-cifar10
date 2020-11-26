def MileStoneLR_WarmUp(optimizer, ep_from_1, gamma, warmup_till, warmup_mode, milsestones):
    if ep_from_1 < warmup_till:
        if warmup_mode == 'linear':
            mul = float(ep_from_1) / float(warmup_till)
        elif warmup_mode == 'const':
            mul = 0.1
        else:
            raise NotImplementedError

    else:
        i = 0
        for i, m in enumerate(milsestones):
            if ep_from_1 <= m:
                break

        if ep_from_1 > milsestones[-1]:
            i += 1

        mul = gamma ** i

    for p in optimizer.param_groups:
        p['lr'] = p['initial_lr'] * mul


def get_lr_strategy(optimizer, milestones=None, gamma=0.5, warmup_till=1, warmup_mode='linear'):
    if milestones is None:
        milestones = [75, 150]

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    return lambda e: MileStoneLR_WarmUp(optimizer, e, gamma, warmup_till, warmup_mode, milestones)
