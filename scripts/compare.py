#!/usr/bin/env python

import os
import configargparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import get_history_log, csv_list, moving_average
import architectures
from architectures import CNN

def filter_hist(h, min_it, max_it, min_y, max_y):
    its = [i for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it]
    vs  = [min(max(v, min_y), max_y) for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it]

    return vs, its

def parse_what_dict(specs):
    what_dict = {}
    for spec in specs.split(';'):
        l,r = spec.split(':')
        if r is None:
            raise ValueError('wrong spec: %s' % spec)
        k,fn = r.split(',')
        if fn is None or fn not in ['min', 'max', 'mean']:
            raise ValueError('wrong spec: %s' % spec)
        what_dict[l] = (k, fn)
    return what_dict

parser = configargparse.ArgumentParser(description='compare multiple histories')
parser.add('--config', '-c', is_config_file=True, help='config file path [default: %(default)s])')
parser.add('files', nargs='+', type=str, help='comma-delimited list of checkpoint files')
parser.add('--out', '-o', help='file name for output image', default='compare.png')
parser.add('--crit-dict', '-d', help='dictionary of logged measures: abbrev:loggged_key,agg_func;...',default='tr:running_train_loss,min;ts:running_valid_loss,min;tr_f:running_loss_last_closure,min;tr_ef:epoch_train_loss,min;w:running_wt,mean;ts_iou:running_valid_iou,max;tr_iou:epoch_train_iou,max;tr_e:epoch_train_loss,min;lr:lr,mean;grad:running_grad,mean')
parser.add('--what', '-w', type=csv_list, default='tr,tr_f,ts', help='list of plot types (tr, ts, tr_f, w, iou)')
parser.add('--grad', '-g', type=int, default=1, help='plot gradients?')
parser.add('--max-iter', '-M', default=100000, type=int, help='maximum iteration')
parser.add('--min-iter', '-m', default=-1, type=int, help='minimum iteration')
parser.add('--max-y',  default=100000, type=float, help='maximum value to include')
parser.add('--min-y',  default=-1, type=float, help='minimum value to include')


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
    if not k in what_dict:
        raise ValueError('unknown history type: %s' % k)

# collect histories

hist = {}

for fname in args.files:
    if not os.path.isfile(fname):
        if not os.path.isdir(fname):
            raise ValueError('checkpoint not found: %s', fname)

        exp_name = filter(lambda x: len(x) > 0, fname.split('/'))[-1]
        guess = os.path.join(fname, 'model_save_%s.pth.tar' % exp_name)
        if not os.path.isfile(guess):
            raise ValueError('checkpoint not found: %s', fname)
        fname = guess

    checkpoint = torch.load(fname, map_location='cpu') # always load to cpu first!
    exp_name = checkpoint['global_state']['args'].experiment
    hist[exp_name] = checkpoint['log']
    batch_size = checkpoint['global_state']['args'].batch_size
    # rescale iterations by batch size
    for entry in hist[exp_name]:
        entry['it'] *= batch_size

# sanity check for what_dict
for k in what_dict:
    found = False
    for exp_name in hist:
        try:
            h = get_history_log(what_dict[k][0], hist[exp_name])
            if len(h[0]) > 0:
                found = True
                break
        except:
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
            h = get_history_log(what_dict[k][0], hist[exp_name])
            n = min(5, len(h[0]))
            m = moving_average(h[0], n)
            best = 0.0
            if what_dict[k][1] == 'min':
                best = min(m)
            elif what_dict[k][1] == 'max':
                best = max(m)
            else:
                best = mean(m)
            v,i = filter_hist(h, args.min_iter, args.max_iter, args.min_y, args.max_y)
            ax0.plot(i, v, colors[c], label='%s %s [%.3f]' % (k, exp_name, best))
            c = (c + 1) % len(colors)
        except:
            pass

if args.grad > 0:
    c = 0
    for exp_name in hist:

        try:
            v,i = filter_hist(get_history_log('running_grad', hist[exp_name]), args.min_iter, args.max_iter, args.min_y, args.max_y)
            ax[1].plot(i, v, colors[c], label='running_grad ' + exp_name)
            c = (c + 1) % len(colors)
        except:
            pass


    ax[1].grid(True, 'both')
    ax[1].legend()

ax0.grid(True, 'both')
ax0.legend()
plt.tight_layout()
fig.savefig(args.out)
plt.close(fig)
