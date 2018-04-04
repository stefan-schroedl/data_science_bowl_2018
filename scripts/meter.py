
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, obj, n=1):
        if isinstance(obj, AverageMeter):
            # add stats from other object
            self.val = obj.val
            self.sum += obj.sum * n
            self.count += obj.count * n
        else:
            self.val = obj
            self.sum += obj * n
            self.count += n

        self.avg = self.sum / self.count

