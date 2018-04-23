from __future__ import division

import math
from collections import defaultdict

def is_number(s):
    try:
        float(s) # for int, long and float
    except:
        return False
    return True


class Meter(object):
    """Computes and stores the average, standard deviation, min, max, and current value of a sequence of numbers, numpy arrays, or torch tensors."""


    def __init__(self):
        self.reset()


    def reset(self):
        self.count = 0
        self.last = float('nan')
        self.min = float('inf')
        self.max = -float('inf')
        self.sum = 0.0
        self.sum2 = 0.0


    def __str__(self):
        return 'Meter(count={}, last={}, min={}, max={}, sum={}, sum2={})'.format(str(self.count), str(self.last), str(self.min), str(self.max), str(self.sum), str(self.sum2))


    def is_valid_element(self, x):
        if is_number(x):
            return True
        typename = type(x).__name__
        return typename == 'ndarray' or typename.endswith('Tensor')


    def update(self, obj, n=1):

        if isinstance(obj, Meter):
            # add stats from other object
            # note: try to preserve types!
            self.last = obj.last
            if self.count == 0:
                self.count = obj.count * n
                self.min = obj.min
                self.max = obj.max
                self.sum = obj.sum * n
                self.sum2 = obj.sum2 * n
            else:
                self.count += obj.count * n
                self.min = min(self.min, obj.min)
                self.max = max(self.max, obj.max)
                self.sum += obj.sum * n
                self.sum2 += obj.sum2 * n

        elif self.is_valid_element(obj):
            self.last = obj
            if self.count == 0:
                self.count = n
                if is_number(obj):
                    self.min = obj
                    self.max = obj
                self.sum = obj * n
                self.sum2 = obj * obj * n
            else:
                self.count += n
                if is_number(obj):
                    self.min = min(self.min, obj)
                    self.max = max(self.max, obj)
                self.sum += obj * n
                self.sum2 += obj * obj * n
        else:
            raise ValueError('wrong type in Meter.update(): %s' % str(obj))


    @property
    def avg(self):
        if self.count == 0:
            return float('nan')
        return self.sum / self.count


    @property
    def std(self):
        if self.count < 2:
            return float('nan')
        try:
            return math.sqrt(max(0.0, (self.sum2  - self.sum * self.sum / self.count)) / (self.count - 1))
        except:
            return float('nan')


class NamedMeter(defaultdict):

    def __init__(self):
        super(NamedMeter, self).__init__(Meter)


    def __str__(self):
        s = 'NamedMeter(\n'
        for k,v in self.items():
            s += '  * ' + str(k) + '\n     ' + str(v) + '\n'
        s += ')'
        return s


    def update(self, key, val=None, n=1):
        if key is None:
            raise ValueError('key cannot be None')
        if val is None:
            if isinstance(key, (dict, NamedMeter)):
                # add stats from other object
                for name in key.keys():
                    self[name].update(key[name], n)
            else:
                raise ValueError('either another NamedMeter, a dictionary, or both key and value needed')
        else:
            self[key].update(val, n)


    def reset(self):
        for v in self.values():
            v.reset()


if __name__ == '__main__':
    m = NamedMeter()
    m.update('a', 1)
    m.update('a', 2)
    m.update('b', 1)

    m2 = NamedMeter()
    m2.update('a', 3)
    m2.update('a', 4)
    m2.update('c', 1)

    m.update(m2)

    print m['a'].avg

