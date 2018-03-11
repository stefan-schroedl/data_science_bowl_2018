#!//usr/bin/env python

import configargparse
import os
import shutil
import copy
from tqdm import tqdm
import timeit
import time

import numpy as np

import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

import transform
from transform import random_rotate90_transform2
import dataset
from dataset import NucleusDataset
import architectures
from architectures import CNN
from utils import mkdir_p, csv_list, strip_end

import matplotlib.pyplot as plt
import operator
from operator import itemgetter

def save_plot(stats, fname):

    xs = map(itemgetter(0), stats)
    ys = map(itemgetter(1), stats)
    zs = map(itemgetter(2), stats)

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(xs,ys,'g')
    ax.plot(xs,zs,'r')
    fig.savefig(fname)
    plt.close(fig)

    
# https://stackoverflow.com/questions/24812253/how-can-i-capture-return-value-with-python-timeit-module
def _template_func(setup, func):
    """Create a timer function. Used if the "statement" is a callable."""
    def inner(_it, _timer, _func=func):
        setup()
        _t0 = _timer()
        for _i in _it:
            retval = _func()
        _t1 = _timer()
        return _t1 - _t0, retval
    return inner

timeit._template_func = _template_func

parser = configargparse.ArgumentParser(description='Data Science Bowl 2018')

parser.add('--config', '-c', default='default.cfg', is_config_file=True, help='config file path [default: %(default)s])')
parser.add('--data', '-d', metavar='DIR', default='/Users/stefan/Documents/nucleus/input/',
           help='path to dataset')
parser.add('--experiment', '-e', required=True, help='experiment name')
parser.add('--out_dir', '-o', help='output directory')
parser.add('--stage', '-s', default='stage1',
           help='stage [default: %(default)s]')
#parser.add('--arch', '-a', metavar='ARCH', default='resnet18',
#                                        choices=model_names,
#                                        help='model architecture: ' +
#                                            ' | '.join(model_names) +
#                                            ' (default: resnet18)')
parser.add('-j', '--workers', default=4, type=int, metavar='N',
           help='number of data loading workers [default: %(default)s]')
parser.add('--epochs', default=1, type=int, metavar='N',
           help='number of total epochs to run [default: %(default)s]')
parser.add('--start-epoch', default=0, type=int, metavar='N',
           help='manual epoch number (useful on restarts)')
parser.add('-b', '--batch-size', default=256, type=int,
           metavar='N', help='mini-batch size (default: 256)')
parser.add('--lr', '--learning-rate', default=0.001, type=float,
           metavar='LR', help='initial learning rate [default: %(default)s]')
parser.add('--momentum', '-m', default=0.9, type=float, metavar='M',
           help='momentum [default: %(default)s]')
parser.add('--weight-decay', '--wd', default=1e-4, type=float,
           metavar='W', help='weight decay [default: %(default)s]')
parser.add('--valid-fraction', '-v', default=0.25, type=float,
           help='validation set fraction [default: %(default)s]')
parser.add('--print-every', '-p', default=10, type=int,
           metavar='N', help='print frequency [default: %(default)s]')
parser.add('--save-every', '-S', default=10, type=int,
           metavar='N', help='save frequency [default: %(default)s]')
parser.add('--eval-every', default=10, type=int,
           metavar='N', help='eval frequency [default: %(default)s]')
parser.add('--resume', default='', type=str, metavar='PATH',
           help='path to latest checkpoint [default: %(default)s]')
parser.add('--override-model-opts', type=csv_list, default='override_model_opts,resume,save_every',
           help='when resuming, change these options [default: %(default)s]')
parser.add('--evaluate', '-E', dest='evaluate', action='store_true',
           help='evaluate model on validation set')

def save_checkpoint(
        model,
        optimizer,
        args,
        stats,
        it,
        is_best=False,
        fname='model_save.pth.tar'):
    m_state_dict = model.state_dict()
    o_state_dict = optimizer.state_dict()

    torch.save({
        'args':args,
        'it': it,
        'model_state_dict': m_state_dict,
        'optimizer_state_dict': o_state_dict,
        'stats': stats},
        fname)
    if is_best:
        print('new best: ', stats[-1])
        pref = strip_end(fname, '.pth.tar')
        shutil.copyfile(fname, '%s_best.pth.tar' % pref)


def load_checkpoint(model, optimizer, args, fname='model_best.pth.tar'):

    if not os.path.isfile(fname):
        error('checkpoint not found: %s', fname)
        
    checkpoint = torch.load(fname)
    it = checkpoint['it']
    stats = checkpoint['stats']
        
    if model:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args:
        override = args.override_model_opts
        
        old_args = checkpoint['args']
        new_args = copy.deepcopy(old_args)

        for k in override:
            if k in args.__dict__:
                v_new = args.__dict__[k]
                if k in old_args.__dict__:
                    v_old = old_args.__dict__[k]
                    if v_old != v_new:
                        print 'WARNING: overriding option %s, old = %s, new = %s' % (k, v_old, v_new)
                new_args.__dict__[k] = v_new
                

    print(
        "=> loaded checkpoint '{}' (iteration {})".format(
            fname, checkpoint['it']))
    return it, stats, new_args


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    cnt = 0
    for i, (img, (labels, labels_seg)) in enumerate(loader):
        img, labels_seg = Variable(img), Variable(labels_seg)
        outputs = model(img)
        loss = criterion(outputs, labels_seg)
        running_loss += loss.data[0]
        cnt = cnt + 1
    l = running_loss / cnt
    return l


def train(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        stats,
        epoch,
        eval_every,
        print_every,
        save_every):
    running_loss = 0.0
    cnt = 0
    global it, best_it, best_loss, args
    for it, (img, (labels, labels_seg)) in tqdm(enumerate(train_loader, it + 1)):
        img, labels_seg = Variable(img), Variable(labels_seg)
        
        model.train(True)

        outputs = model(img)
        loss = criterion(outputs, labels_seg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt = cnt + 1

        running_loss += loss.data[0]
        if cnt > 0 and it % eval_every == 0:

            l = validate(model, valid_loader, criterion)
            stats.append((it, running_loss / cnt, l))
            running_loss = 0.0

            if cnt > 0 and it % print_every == 0:
                print('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, it, stats[-1][1], stats[-1][2]))
                save_plot(stats, os.path.join(args.out_dir, 'progress.png'))

            if it % save_every == 0:
                is_best = False
                if best_loss > l:
                    best_loss = l
                    best_it = it
                    is_best = True
                    save_checkpoint(
                        model,
                        optimizer,
                        args,
                        stats,
                        it,
                        is_best,
                        os.path.join(args.out_dir, 'model_save_%s.pth.tar' % args.experiment))
            cnt = 0
    return it, best_loss, best_it

def baseline(
        train_loader,
        valid_loader,
        criterion,
        it):
    
    m = 0.0
    cnt = 0.0
    for i, (img, (labels, labels_seg)) in enumerate(train_loader):
        if i > it:
            break
        m += labels_seg[0].sum()
        cnt += labels_seg[0].numel()
    m = m / cnt

    running_loss = 0.0
    
    cnt = 0
    for i, (img, (labels,labels_seg)) in enumerate(valid_loader):
        outputs = labels_seg.clone()        
        outputs = torch.clamp(outputs, m, m)
        img, labels = Variable(img), Variable(labels_seg)
        outputs = Variable(outputs)
    
        loss = criterion(outputs, labels_seg)

        running_loss += loss.data[0]
        cnt += 1

    return running_loss/cnt, m


def adjust_learning_rate(optimizer, epoch, lr0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#Note: for image and mask, there is no compatible solution that can use transforms.Compse(), see https://github.com/pytorch/vision/issues/9
#transformations = transforms.Compose([random_rotate90_transform2(),transforms.ToTensor(),])

def train_transform(img, mask, mask_seg):
    # HACK (make dims consistent, first one is channels)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 2)
    if len(mask_seg.shape) == 2:
        mask_seg = np.expand_dims(mask_seg, 2)
    img, mask, mask_seg = random_rotate90_transform2(0.5, img, mask, mask_seg)
    img = ToTensor()(img)
    mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
    mask_seg = torch.from_numpy(np.transpose(mask_seg, (2, 0, 1))).float()
    return img, mask, mask_seg


#NucleusDataset(args.data, args.stage, transform=train_transform)


def main():
    global it, best_it, best_loss, stats, args
    args = parser.parse_args()

    if args.out_dir is None:
       args.out_dir = 'experiments/%s' % args.experiment 
    mkdir_p(args.out_dir)

    # create model
    model = CNN()
    it = 0
    best_loss = 1e20
    best_it = 0
    stats = []
                    
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), args.lr,
                           #momentum=args.momentum,
                           weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        it, stats, args = load_checkpoint(model, optimizer, args, args.resume)

    # Data loading
    print 'loading data'
    #dset = NucleusDataset(args.data, args.stage, transform=train_transform)
    def load_data():
        return NucleusDataset(args.data, args.stage, transform=train_transform)
    timer = timeit.Timer(load_data)
    t,dset = timer.timeit(number=1)
    print 'load time', t

    # hack: this image format (1388, 1040) occurs only ones, stratify complains ..
    dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

    stratify = dset.data_df['images'].map(lambda x: '{}'.format(x.size))
    train_dset, valid_dset = dset.train_test_split(test_size=args.valid_fraction, random_state=1, shuffle=True, stratify=stratify)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=1, shuffle=True)

    if args.evaluate:
        validate(model, valid_loader, criterion)
        return

    if args.resume:
        l = validate(model, valid_loader, criterion)
        print 'validation for loaded model: ', l 
        
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        it, best_loss, best_it = train(train_loader, valid_loader, model, criterion, optimizer, stats, epoch, args.eval_every, args.print_every, args.save_every)
    print it, best_loss, best_it


if __name__ == '__main__':
    main()
