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
    its = [i for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it and v >= min_y and v <= max_y]
    vs  = [v for v,i in zip(h[0], h[1]) if i >= min_it and i <= max_it and v >= min_y and v <= max_y]
    return vs, its

parser = configargparse.ArgumentParser(description='compare mutliple graphs')

parser.add('files', nargs='+', type=str, help='comma-delimited list of checkpoint files')
parser.add('--out', '-o', help='file name for output image', default='compare.png')
parser.add('--what', '-w', type=csv_list, default='tr,tr_f,ts', help='plot train, train final, or test')
parser.add('--grad', '-g', type=int, default=1, help='plot gradients?')
parser.add('--max-iter', '-M', default=100000, type=int, help='maximum iteration')
parser.add('--min-iter', '-m', default=-1, type=int, help='minimum iteration')
parser.add('--max-y',  default=100000, type=int, help='maximum value to include')
parser.add('--min-y',  default=-1, type=int, help='minimum value to include')


args = parser.parse_args()
    
model = architectures.CNN()
tr = []
ts = []
gr = []
exps = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# collect histories

for fname in args.files:
    if not os.path.isfile(fname):
        #if os.path.isdir(fname):
        #    # try to find latest checkpoint
        #    for f in glob.glob(os.path.join(fname, ''):
        #        print f     
        raise ValueError('checkpoint not found: %s', fname)
    checkpoint = torch.load(fname)
    tr.append(filter_hist(get_history_log('train_loss', checkpoint['log']), args.min_iter, args.max_iter, args.min_y, args.max_y))
    ts.append(filter_hist(get_history_log('valid_loss', checkpoint['log']), args.min_iter, args.max_iter, args.min_y, args.max_y))
    tr_f = None
    try:
        tr_f.append(filter_hist(get_history_log('final_train_loss', checkpoint['log']), args.min_iter, args.max_iter, args.min_y, args.max_y))
    except:
        pass
    gr.append(filter_hist(get_history_log('grad', checkpoint['log']), args.min_iter, args.max_iter, args.min_y, args.max_y))
    exps.append(checkpoint['global_state']['args'].experiment)

# prepare plot
if args.grad > 0:
    fig, ax = plt.subplots(2, 1)
    ax0 = ax[0]
else:
    fig, ax = plt.subplots(1, 1)
    ax0 = ax

c = 0
for i in range(len(exps)):
    if 'tr' in args.what:
        ax0.plot(tr[i][1], tr[i][0], colors[c], label='tr ' + exps[i])
        c = (c + 1) % len(colors)
    if 'ts' in args.what:
        ax0.plot(ts[i][1], ts[i][0], colors[c], label='ts ' + exps[i])
        c = (c + 1) % len(colors)
    if 'tr_f' in args.what and tr_f is not None:
        ax0.plot(ts[i][1], tr_f[i][0], colors[c], label='ts ' + exps[i])
        c = (c + 1) % len(colors)

c = 0
if args.grad > 0:
    for i in range(len(exps)):
        ax[1].plot(gr[i][1], gr[i][0], colors[c], label='grad ' + exps[i])
        c = (c + 1) % len(colors)
    ax[1].grid(True, 'both')
    ax[1].legend()

ax0.grid(True, 'both')
ax0.legend()
plt.tight_layout()
fig.savefig(args.out)
plt.close(fig)
