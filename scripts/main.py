#!//usr/bin/env python

import configargparse
import os
import shutil
import copy
from tqdm import tqdm
import timeit
import time
import logging

import numpy as np

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

import matplotlib.pyplot as plt
import operator
from operator import itemgetter


import transform
from transform import random_rotate90_transform2
import dataset
from dataset import NucleusDataset
import architectures
from architectures import CNN
from utils import mkdir_p, csv_list, strip_end, init_logging, get_log, set_log, clear_log, insert_log, get_latest_log, get_history_log
from adjust_learn_rate import get_learning_rate



import post_process
from post_process import parametric_pipeline
from loss import iou_metric, diagnose_errors, show_compare_gt

def get_checkpoint_file(args):
    return os.path.join(args.out_dir, 'model_save_%s.pth.tar' % args.experiment)

def save_plot(fname):
    train_loss, train_loss_it = get_history_log('train_loss')
    valid_loss, valid_loss_it = get_history_log('valid_loss')

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(train_loss_it, train_loss, 'g')
    ax.plot(valid_loss_it, valid_loss, 'r')
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
parser.add('--patience', default=10, type=int,
           metavar='N', help='patience for lr scheduler [default: %(default)s]')
parser.add('--cooldown', default=5, type=int,
           metavar='N', help='cooldown for lr scheduler [default: %(default)s]')
parser.add('--min_lr', default=0.00001, type=float,
           metavar='N', help='minimum learn rate for scheduler [default: %(default)s]')
parser.add('--resume', default='', type=str, metavar='PATH',
           help='path to latest checkpoint [default: %(default)s]')
parser.add('--override-model-opts', type=csv_list, default='override_model_opts,resume,save_every',
           help='when resuming, change these options [default: %(default)s]')
parser.add('--evaluate', '-E', dest='evaluate', action='store_true',
           help='evaluate model on validation set')
parser.add('--calc-iou', type=int, default=0, help='calculate iou and exit')
parser.add('--random-seed', type=int, default=2018, help='set random number generator seed [default: %(default)s]')
parser.add('--verbose', '-V', action='store_true', help='verbose logging')
parser.add('--force-overwrite', type=int, default=0, help='overwrite existing checkpoint, if it exists')
parser.add('--log-file', help='write logging output to file')

def save_checkpoint(
        model,
        optimizer,
        args,
        it,
        is_best=False,
        fname='model_save.pth.tar'):

    global best_loss, best_it
    m_state_dict = model.state_dict()
    o_state_dict = optimizer.state_dict()

    torch.save({
        'args':args,
        'it': it,
        'model_state_dict': m_state_dict,
        'optimizer_state_dict': o_state_dict,
        'best_loss': best_loss,
        'best_it': best_it,
        'log': get_log()},
        fname)
    if is_best:
        logging.info('new best: it = %d, train = %.5f, valid = %.5f' % (get_latest_log('it'), get_latest_log('train_loss'), get_latest_log('valid_loss')))
        pref = strip_end(fname, '.pth.tar')
        shutil.copyfile(fname, '%s_best.pth.tar' % pref)


def load_checkpoint(model, optimizer, args, fname='model_best.pth.tar'):

    if not os.path.isfile(fname):
        raise ValueError('checkpoint not found: %s', fname)
        
    checkpoint = torch.load(fname)
    it = checkpoint['it']
    set_log(checkpoint['log'])
    best_it = checkpoint['best_it']
    best_loss = checkpoint['best_loss']

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
                        logging.warn(' overriding option %s, old = %s, new = %s' % (k, v_old, v_new))
                new_args.__dict__[k] = v_new

        # copy new options not present in saved file
        for k in args.__dict__:
            if k not in old_args.__dict__:
                new_args.__dict__[k] = args.__dict__[k]

    logging.info(
        "=> loaded checkpoint '{}' (iteration {})".format(
            fname, checkpoint['it']))
    return it, new_args


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


def compute_iou(model, loader):
    model.eval()
    pred = [(model(Variable(img,requires_grad=False)).data.numpy().squeeze(), mask.numpy().squeeze()) for img, (mask,mask_seg) in tqdm(iter(loader))]
    img_th = [(parametric_pipeline(img, circle_size=4), mask) for img, mask in pred]
    ious = [iou_metric(i,m) for (i,m) in img_th]
    msg = 'iou: mean = %.5f, med = %.5f' % (np.mean(ious), np. median(ious))
    print msg
    logging.info(msg)


def train(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        eval_every,
        print_every,
        save_every):
    running_loss = 0.0
    cnt = 0
    global it, best_it, best_loss, args, lr
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
            train_loss = running_loss / cnt
            insert_log(it, 'train_loss', train_loss)
            insert_log(it, 'valid_loss', l)
            running_loss = 0.0

            if cnt > 0 and it % print_every == 0:
                logging.info('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, it, train_loss, l))
                save_plot(os.path.join(args.out_dir, 'progress.png'))

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
                        it,
                        is_best,
                        get_checkpoint_file(args))
            cnt = 0

            scheduler.step(train_loss)

            #lr_new = lr_scheduler.on_epoch_end(it, lr, train_loss)
            #if lr_new != lr:
            #    lr = lr_new
            #    adjust_learning_rate(scheduler, lr)
            lr_new = get_learning_rate(scheduler.optimizer)
            if lr_new != lr:
                logging.info('[%d, %d]\tlearning rate changed from %f to %f' % (epoch, it, lr, lr_new))
                lr = lr_new
            insert_log(it, 'lr', get_learning_rate(scheduler.optimizer))

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
    global it, best_it, best_loss, LOG, args, lr
    args = parser.parse_args()

    if args.out_dir is None:
       args.out_dir = 'experiments/%s' % args.experiment 
    mkdir_p(args.out_dir)

    # for later info, save the current configuration
    if args.config is not None and os.path.isfile(args.config):
        shutil.copy(args.config, args.out_dir)

    if args.log_file is None:
        args.log_file = os.path.join(args.out_dir, '%s.log' % args.experiment)

    init_logging(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    # create model
    model = CNN()
    it = 0
    best_loss = 1e20
    best_it = 0
     
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr,
                           #momentum=args.momentum,
                           weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        it, args = load_checkpoint(model, optimizer, args, args.resume)
        args.force_overwrite = 1
    else:
        # prevent accidental overwriting
        ckpt_file = get_checkpoint_file(args)
        if os.path.isfile(ckpt_file) and args.force_overwrite == 0:
            raise ValueError('checkpoint already exists, exiting: %s' % ckpt_file)
        clear_log()

    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, cooldown=args.cooldown, min_lr=args.min_lr, verbose=1)

    # Data loading
    logging.info('loading data')
    #dset = NucleusDataset(args.data, args.stage, transform=train_transform)
    def load_data():
        return NucleusDataset(args.data, args.stage, transform=train_transform)
    timer = timeit.Timer(load_data)
    t,dset = timer.timeit(number=1)
    logging.info('load time: %.1f' % t)

    # hack: this image format (1388, 1040) occurs only ones, stratify complains ..
    dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

    stratify = dset.data_df['images'].map(lambda x: '{}'.format(x.size))
    train_dset, valid_dset = dset.train_test_split(test_size=args.valid_fraction, random_state=1, shuffle=True, stratify=stratify)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=1, shuffle=True)

    if args.calc_iou > 0:
        compute_iou(model, train_loader)
        return

    if args.evaluate:
        validate(model, valid_loader, criterion)
        return

    if args.resume:
        l = validate(model, valid_loader, criterion)
        logging.info('validation for loaded model: %s' % l)



    for epoch in range(args.epochs):
        it, best_loss, best_it = train(train_loader, valid_loader, model, criterion, optimizer, scheduler, epoch, args.eval_every, args.print_every, args.save_every)
        logging.info('final best: it = %d, valid = %.5f' % (best_it, best_loss))


if __name__ == '__main__':
    main()
