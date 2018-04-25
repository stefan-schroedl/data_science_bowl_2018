#!/usr/bin/env python

import sys
import os
import configargparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import get_log, list_log_keys, csv_list, moving_average, checkpoint_file_from_dir
import architectures


def filter_hist(h, min_it, max_it, min_y, max_y):
    its = [i for v, i in zip(h[0], h[1]) if i >= min_it and i <= max_it]
    vs = [min(max(v, min_y), max_y)
          for v, i in zip(h[0], h[1]) if i >= min_it and i <= max_it]

    return vs, its


def parse_what_dict(specs):
    what_dict = {}
    for spec in specs.split(';'):
        l, r = spec.split(':')
        if r is None:
            raise ValueError('wrong spec: %s' % spec)
        k, fn = r.split(',')
        if fn is None or fn not in ['min', 'max', 'mean']:
            raise ValueError('wrong spec: %s' % spec)
        what_dict[l] = (k, fn)
    return what_dict


crit_default = ('tr:train_last_loss,min;'
                'tr_e:train_avg_loss,min;'
                'ts:valid_avg_loss,min;'
                'w:train_avg_inst_wt,mean;'
                'ts_iou:valid_avg_iou,max;'
                'tr_iou:train_avg_iou,max;'
                'tr_seg:train_avg_seg,min;'
                'ts_seg:valid_avg_seg,min;'
                'tr_cont:train_avg_cont,min;'
                'ts_cont:valid_avg_cont,min;'
                'lr:lr,min;'
                'grad:train_avg_grad,mean')

parser = configargparse.ArgumentParser(
    description='compare multiple histories')
parser.add('--config', '-c', is_config_file=True,
           help='config file path [default: %(default)s])')
parser.add('files', nargs='+', type=str,
    help='comma-delimited list of checkpoint files')
parser.add('--out', '-o', help='file name for output image', default='compare.png')
parser.add('--crit-dict', '-d',
    help='dictionary of logged measures: <abbrev>:<loggged_key>,<agg_func>;...',
    default=crit_default)
parser.add('--what', '-w', type=csv_list, default='tr,tr_e,ts',
           help='list of plot types (tr, ts, tr_f, w, iou)')
parser.add('--grad', '-g', type=int, default=1, help='plot gradients?')
parser.add('--max-iter', '-M', default=100000, type=int, help='maximum iteration')
parser.add('--min-iter', '-m', default=-1, type=int, help='minimum iteration')
parser.add('--max-y', default=100000, type=float, help='maximum value to include')
parser.add('--min-y', default=-1, type=float, help='minimum value to include')
parser.add('--list-keys', '-l', default=0, type=int, help='show keys stored in log')


args = parser.parse_args()

model = architectures.CNN()
tr = []
tr_f = []
w = []
ts = []
gr = []
exps = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

what_dict = parse_what_dict(args.crit_dict)

for k in args.what:
    if k not in what_dict:
        raise ValueError('unknown history type: %s' % k)

# collect histories

hist = OrderedDict()

for fname in args.files:
    fname = checkpoint_file_from_dir(fname)
    # always load to cpu first!
    checkpoint = torch.load(fname, map_location='cpu')
    exp_name = checkpoint['global_state']['args'].experiment
    hist[exp_name] = checkpoint['log']
    batch_size = checkpoint['global_state']['args'].batch_size
    # rescale iterations by batch size
    for entry in hist[exp_name]:
        entry['it'] *= batch_size
    if args.list_keys > 0:
        keys = list_log_keys(hist[exp_name])
        print 'keys in file %s (experiment "%s"):' % (fname, exp_name)
        for k in keys:
            print '    %s' % k

if args.list_keys > 0:
    sys.exit(0)

# sanity check for what_dict
for k in what_dict:
    found = False
    for exp_name in hist:
        try:
            h = get_log(what_dict[k][0], hist[exp_name])
            if len(h[0]) > 0:
                found = True
                break
        except BaseException:
            pass
    if not found:
        print 'WARNING: key %s=%s configured but not found in any history' % (k, what_dict[k][0])

# prepare plot
if args.grad > 0:
    fig, ax = plt.subplots(2, 1)
    ax0 = ax[0]
else:
    fig, ax = plt.subplots(1, 1)
    ax0 = ax

c = 0
for exp_name in hist:
    for k in args.what:
        try:
            h = get_log(what_dict[k][0], hist[exp_name])
            n = min(5, len(h[0]))
            m = moving_average(h[0], n)
            best = 0.0
            if what_dict[k][1] == 'min':
                best = min(m)
            elif what_dict[k][1] == 'max':
                best = max(m)
            else:
                best = np.mean(m)
            v, i = filter_hist(
                h, args.min_iter, args.max_iter, args.min_y, args.max_y)

            ax0.plot(
                i, v, colors[c], label='%s %s [%.3g]' %
                (k, exp_name, best))
            c = (c + 1) % len(colors)
        except BaseException:
            pass

if args.grad > 0:
    c = 0
    for exp_name in hist:

        try:
            v, i = filter_hist(get_log(
                    'train_avg_grad', hist[exp_name]), args.min_iter, args.max_iter, args.min_y, args.max_y)
            ax[1].plot(i, v, colors[c], label='grad ' + exp_name)
            c = (c + 1) % len(colors)
        except BaseException:
            pass

    ax[1].grid(True, 'both')
    ax[1].legend()

ax0.grid(True, 'both')
ax0.legend()
plt.tight_layout()
fig.savefig(args.out)
plt.close(fig)
