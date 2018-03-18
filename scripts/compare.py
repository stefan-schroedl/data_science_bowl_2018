#!/usr/bin/env python

import os
import configargparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import get_history_log, csv_list
import architectures
from architectures import CNN

def filter_hist(h, min_it, max_it, min_y, max_y):
    #its = [i for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it and v >= min_y and v <= max_y]
    #vs  = [v for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it and v >= min_y and v <= max_y]
    its = [i for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it]
    vs  = [min(max(v, min_y), max_y) for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it]

    return vs, its

parser = configargparse.ArgumentParser(description='compare mutliple graphs')

parser.add('files', nargs='+', type=str, help='comma-delimited list of checkpoint files')
parser.add('--out', '-o', help='file name for output image', default='compare.png')
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

what_dict = {'tr':'train_loss', 'ts':'valid_loss', 'tr_f':'final_train_loss', 'w':'bp_wt', 'iou':'valid_iou'}

for k in args.what:
    if not k in what_dict:
        raise ValueError('unknown history type: %s' % k)

# collect histories

hist = {}

for fname in args.files:
    if not os.path.isfile(fname):
        #if os.path.isdir(fname):
        #    # try to find latest checkpoint
        #    for f in glob.glob(os.path.join(fname, ''):
        #        print f     
        raise ValueError('checkpoint not found: %s', fname)
    checkpoint = torch.load(fname)
    hist[checkpoint['global_state']['args'].experiment] =  checkpoint['log']


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
        if k == 'grad':
            next
        try:
            v,i = filter_hist(get_history_log(what_dict[k], hist[exp_name]), args.min_iter, args.max_iter, args.min_y, args.max_y)
            ax0.plot(i, v, colors[c], label=k + ' ' + exp_name)
            c = (c + 1) % len(colors)
        except:
            pass

if args.grad > 0:
    c = 0
    for exp_name in hist:

        try:
            v,i = filter_hist(get_history_log('grad', hist[exp_name]), args.min_iter, args.max_iter, args.min_y, args.max_y)
            ax[1].plot(i, v, colors[c], label='grad ' + exp_name)
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
