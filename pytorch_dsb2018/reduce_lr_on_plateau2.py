import logging
import torch
import math
from bisect import bisect_right
from torch.optim.optimizer import Optimizer

class ReduceLROnPlateau2(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    This class is a variation of torch.optim.lr_scheduler.ReduceLROnPlateau.
    The difference is to wait with the learn rate reduction until the
    deterioration is again smaller than patience_threshold times the maximum
    deterioration since the last optimum. Error sometimes tends to oscillate
    between large deviations and some close to the original best one.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        patience_threshold (float): Wait to reduce the learn rate until
            at most that much of the observed degradation. Default: .1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 patience_threshold=0.1, verbose=False, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.patience_threshold = patience_threshold
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.worst_after_bst = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.is_acceptable_degradation = None
        self.eps = eps
        self.last_epoch = -1
        # important difference to ReduceLROnPlateau:
        # wait until error oscillates back towards optimum before reducing the learning rate
        self.last = None
        self.waiting_to_reduce = False
        # important difference to ReduceLROnPlateau:
        # while the error is going down immediately after reduction, don't start cooldown yet!
        self.waiting_for_deterioration = False
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode, patience_threshold=patience_threshold)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.worst_after_best = - self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.waiting_to_reduce = False
        self.waiting_for_deterioration = False

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.verbose:
            logging.info('[epoch {:5d}] scheduler: num_bad_epochs = {}, current = {:.4g}, best = {:.4g}, worst_after_best={:.4g}, cooldown = {}, waiting_to_reduce = {}, waiting_for_deterioration = {}'.format(epoch, self.num_bad_epochs, current, self.best, self.worst_after_best, self.in_cooldown, self.waiting_to_reduce, self.waiting_for_deterioration))

        if not self.waiting_to_reduce and self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.worst_after_best = self.best
        else:
            self.num_bad_epochs += 1
            if self.mode == 'min':
                self.worst_after_best = max(self.worst_after_best, current)
            else:
                self.worst_after_best = min(self.worst_after_best, current)

        is_not_improving = self.last is None or self.last == current or self.is_better(self.last, current)
        if is_not_improving:
            self.waiting_for_deterioration = False
        self.last = current

        if self.in_cooldown or self.waiting_for_deterioration:
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
            self.worst_after_best = self.best

        if self.in_cooldown and not self.waiting_for_deterioration:
            self.cooldown_counter -= 1

        if not self.waiting_to_reduce and self.num_bad_epochs > self.patience:
            self.waiting_to_reduce = True
            if self.verbose:
                logging.info('[epoch {:5d}] scheduler: ran out of patience, wating to reduce lr'.format(epoch))

        if self.waiting_to_reduce and self.is_acceptable_degradation(current, self.best, self.worst_after_best):
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.worst_after_best = self.best
            self.waiting_to_reduce = False
            self.waiting_for_deterioration = True

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logging.info('[epoch {}] scheduler: reducing learning rate'
                                 ' of group {} to {:.4g}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode, patience_threshold):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
            self.is_acceptable_degradation = lambda a,best,worst: (a-best) <= patience_threshold*(max(worst-best, best))
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
            self.is_acceptable_degradation = lambda a,best,worst: (a-best) <= patience_threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
            self.is_acceptable_degradation = lambda a,best,worst: (best-a) <= patience_threshold
            self.is_acceptable_degradation = lambda a,best,worst: (best-a) <= patience_threshold*(best-worst)
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')
