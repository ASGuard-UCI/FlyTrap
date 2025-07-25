import math

from torch import optim

from ..builder import OPTIMIZER, SCHEDULER


@OPTIMIZER.register_module()
class Adam(optim.Adam):
    def __init__(self, param, **kwargs):
        super().__init__(param, **kwargs)


@SCHEDULER.register_module()
class WarmupCosineDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        if warmup_steps > 0:
            self.initial_lr = optimizer.param_groups[0]['lr'] / warmup_steps
        else:
            self.initial_lr = optimizer.param_groups[0]['lr']
        self.lr = optimizer.param_groups[0]['lr']
        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch, verbose=False)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.initial_lr * (self.last_epoch + 1)
        else:
            cos_decay = 0.5 * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            lr = (self.lr - self.eta_min) * cos_decay + self.eta_min
        return lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
