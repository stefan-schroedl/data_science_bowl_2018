#!//usr/bin/env python

import sys
import os
import shutil

import configargparse
import copy
import timeit
import time
import logging

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim, nn

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

import cv2

from sklearn.neighbors import KNeighborsClassifier

import utils
from utils import mkdir_p, csv_list, strip_end, init_logging, get_log, set_log, clear_log, insert_log, get_latest_log, get_history_log
from adjust_learn_rate import get_learning_rate

from KNN import *

import transform
from transform import random_rotate90_transform2
import dataset
from dataset import NucleusDataset

import architectures
from architectures import CNN

import post_process
from post_process import parametric_pipeline
import loss
from loss import iou_metric, diagnose_errors, show_compare_gt, union_intersection, precision_at


def get_checkpoint_file(args):
    return os.path.join(args.out_dir, 'model_save_%s.pth.tar' % args.experiment)

def save_plot(fname):
    train_loss, train_loss_it = get_history_log('train_loss')
    valid_loss, valid_loss_it = get_history_log('valid_loss')
    grad, grad_it = get_history_log('grad')

    fig, ax = plt.subplots( nrows=2, ncols=1)
    ax[0].plot(train_loss_it, train_loss, 'g', label='train')
    ax[0].plot(valid_loss_it, valid_loss, 'r', label='test')
    ax[1].plot(grad_it, grad, label='grad')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
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

parser.add('--model', help='cnn/knn', choices=['knn', 'cnn'], required=True, default="")
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
parser.add('--grad-accum', default=1, type=int,
           metavar='N', help='number of batches between gradient descent [default: %(default)s]')
parser.add('--lr', '--learning-rate', default=0.001, type=float,
           metavar='LR', help='initial learning rate [default: %(default)s]')
parser.add('--min_lr', default=0.00001, type=float,
           metavar='N', help='minimum learn rate for scheduler [default: %(default)s]')
parser.add('--momentum', '-m', default=0.9, type=float, metavar='M',
           help='momentum [default: %(default)s]')
parser.add('--weight-decay', default=1e-4, type=float,
           metavar='W', help='weight decay [default: %(default)s]')
parser.add('--use-instance-weights', default=0, type=int,
           metavar='N', help='apply instance weights during training [default: %(default)s]')
parser.add('--clip-gradient', default=0.25, type=float,
           metavar='C', help='clip excessive gradients during training [default: %(default)s]')
parser.add('--criterion', '-C', default='mse', choices=['mse','bce','jaccard','dice'],
           metavar='C', help='loss function [default: %(default)s]')
parser.add('--optim', '-O', default='sgd', choices=['sgd','adam'],
           help='optimization algorithm [default: %(default)s]')
parser.add('--valid-fraction', '-v', default=0.25, type=float,
           help='validation set fraction [default: %(default)s]')
parser.add('--stratify', type=int, default=1, help='stratify train/test split according to image size [default: %(default)s]')
parser.add('--print-every', '-p', default=10, type=int,
           metavar='N', help='print frequency [default: %(default)s]')
parser.add('--save-every', '-S', default=10, type=int,
           metavar='N', help='save frequency [default: %(default)s]')
parser.add('--eval-every', default=10, type=int,
           metavar='N', help='eval frequency [default: %(default)s]')
parser.add('--patience', default=10, type=int,
           metavar='N', help='patience for lr scheduler [default: %(default)s]')
parser.add('--cooldown', default=10, type=int,
           metavar='N', help='cooldown for lr scheduler [default: %(default)s]')
parser.add('--resume', default='', type=str, metavar='PATH',
           help='path to latest checkpoint [default: %(default)s]')
parser.add('--override-model-opts', type=csv_list, default='override-model-opts,resume,save-every',
           help='when resuming, change these options [default: %(default)s]')
parser.add('--evaluate', '-E', dest='evaluate', action='store_true',
           help='evaluate model on validation set')
parser.add('--calc-iou', type=int, default=0, help='calculate iou and exit')
parser.add('--random-seed', type=int, default=2018, help='set random number generator seed [default: %(default)s]')
parser.add('--verbose', '-V', action='store_true', help='verbose logging')
parser.add('--force-overwrite', type=int, default=0, help='overwrite existing checkpoint, if it exists')
parser.add('--log-file', help='write logging output to file')

def save_checkpoint(fname,
                    model,
                    optimizer=None,
                    global_state=None,
                    is_best = False):

    m_state_dict = model.state_dict()
    o_state_dict = optimizer.state_dict()

    s = {'model_state_dict': model.state_dict(),
         'log': get_log()}

    if global_state:
            s['global_state'] = global_state

    if optimizer:
        s['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(s, fname)

    if is_best:
        logging.info('new best: it = %d, train = %.5f, valid = %.5f' % (get_latest_log('it'), get_latest_log('train_loss'), get_latest_log('valid_loss')))
        pref = strip_end(fname, '.pth.tar')
        shutil.copyfile(fname, '%s_best.pth.tar' % pref)


def load_checkpoint(fname,
                    model,
                    optimizer=None,
                    global_state=None):

    if not os.path.isfile(fname):
        raise ValueError('checkpoint not found: %s', fname)

    checkpoint = torch.load(fname)
    try:
        set_log(checkpoint['log'])
    except:
        pass

    if global_state and 'global_state' in checkpoint:
        for k,v in checkpoint['global_state'].iteritems():
            global_state[k] = v

    if model:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if global_state and 'args' in global_state and 'global_state' in checkpoint and 'args' in checkpoint['global_state']:
        args = global_state['args']
        override = args.override_model_opts

        old_args = checkpoint['global_state']['args']
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

        global_state['args'] = new_args

    it = '?'
    if 'global_state' in checkpoint and 'it' in checkpoint['global_state']:
        it = checkpoint['global_state']['it']
    logging.info(
        "=> loaded checkpoint '{}' (iteration {})".format(
            fname, it))

def torch_to_numpy(t):
    return (t.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

def validate_knn(model, loader, criterion):
    running_loss = 0.0
    cnt = 0
    for i, (img, (labels, labels_seg)) in enumerate(loader):
        p_img,p_seg,p_boundary,p_blend=model.predict(img)
        torch_p_seg = torch.from_numpy(p_seg[None,:,:].astype(np.float)/255).float()
        #torch_p_boundary = torch.from_numpy(p_boundary[None,:,:].astype(np.float)/255).float()
        #torch_p_blend = torch.from_numpy(p_blend[None,:,:].astype(np.float)/255).float()
        border=np.full((p_img.shape[0],5,3),255).astype(np.uint8)
        cv2.imshow('img and reconstructed img',np.concatenate((torch_to_numpy(img),border,p_img),axis=1))
        cv2.imshow('seg and reconstructed seg',np.concatenate((torch_to_numpy(labels_seg),border[:,:,:1],p_seg[:,:,None]),axis=1))
        #cv2.imshow('pbound',p_boundary)
        #cv2.imshow('pblend',p_blend)
        cv2.waitKey(10)
        loss = criterion(Variable(torch_p_seg), Variable(labels_seg.float()))
        running_loss += loss.data[0]
        cnt = cnt + 1
    l = running_loss / cnt
    return l

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

def train_knn(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        eval_every,
        print_every,
        save_every,
        global_state):
    running_loss = 0.0
    cnt = 0
    for global_state['it'], (img, (labels, labels_seg)) in tqdm(enumerate(train_loader, global_state['it'] + 1)):
        model.prepare_fit(img,labels,labels_seg)

        cnt = cnt + 1

        if cnt > 0 and global_state['it'] % eval_every == 0:
            model.fit()
            l = validate_knn(model, valid_loader, criterion)
            img,mask,boundary,blend=model.predict(img)
            insert_log(global_state['it'], 'train_loss', running_loss / cnt)
            insert_log(global_state['it'], 'valid_loss', l)
            running_loss = 0.0

            if cnt > 0 and global_state['it'] % print_every == 0:
                print('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, global_state['it'], stats[-1][1], stats[-1][2]))
            if global_state['it'] % save_every == 0:
                is_best = False
                if global_state['best_loss'] > l:
                    global_state['best_loss'] = l
                    global_state['best_it'] = global_state['it']
                    is_best = True
    return global_state['it'], global_state['best_loss'], global_state['best_it']


def compute_iou(model, loader):
    model.eval()
    pred = [(model(Variable(img,requires_grad=False)).data.numpy().squeeze(), mask.numpy().squeeze()) for img, (mask,mask_seg) in tqdm(iter(loader))]
    img_th = [(parametric_pipeline(img, circle_size=4), mask) for img, mask in pred]
    ious = [iou_metric(i,m) for (i,m) in img_th]
    msg = 'iou: mean = %.5f, med = %.5f' % (np.mean(ious), np. median(ious))
    logging.info(msg)
    print msg


def backprop_weight(labels, img, global_state, thresh=0.1):
    img_th = parametric_pipeline(img, circle_size=4)
    union, intersection, area_true, area_pred = union_intersection(labels, img_th)

    # Compute the intersection over union
    iou = intersection.astype(float) / union

    tp, fp, fn, matches_by_pred, matches_by_target = precision_at(iou, thresh)

    w = 1.0

    denom = 1.0 * (tp + fp + fn)

    if tp + fn == 0.0:
        w = 0.0

    if denom > 0.0:
        w = 1.0 / denom

    # normalize with running average
    w = w / (global_state['bp_wt_sum'] / global_state['bp_wt_cnt'])
    global_state['bp_wt_sum'] += w
    global_state['bp_wt_cnt'] += 1

    return w


def train_cnn(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        eval_every,
        print_every,
        save_every,
        global_state):

    running_loss = 0.0
    running_grad = 0.0
    valid_cnt = 0

    # global it, best_it, best_loss, args, lr

    for global_state['it'], (img, (labels, labels_seg)) in tqdm(enumerate(train_loader, global_state['it'] + 1)):

        img, labels_seg = Variable(img), Variable(labels_seg)

        model.train(True)

        outputs = model(img)

        if global_state['args'].use_instance_weights > 0:
            w = backprop_weight(labels.numpy().squeeze(), outputs.data[0].numpy().squeeze(), global_state)
            criterion._buffers['weights'] = torch.FloatTensor([w])

        loss = criterion(outputs, labels_seg)

        # gradient accumulation
        accum_total = global_state['args'].grad_accum
        accum_step = global_state['grad_accum_it']
        if accum_step == 0:
            optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm(model.parameters(), global_state['args'].clip_gradient)
        if accum_step == accum_total - 1:
            optimizer.step()
        accum_step = (accum_step + 1) % accum_total
        global_state['grad_accum_it'] = accum_step

        valid_cnt += 1
        running_loss += loss.data[0]
        running_grad += grad

        if valid_cnt > 0 and global_state['it'] % eval_every == 0:
            l = validate(model, valid_loader, criterion)
            train_loss = running_loss / valid_cnt
            train_grad = running_grad / valid_cnt
            insert_log(global_state['it'], 'train_loss', train_loss)
            insert_log(global_state['it'], 'grad', train_grad)
            insert_log(global_state['it'], 'valid_loss', l)
            running_loss = 0.0
            running_grad = 0.0

            if valid_cnt > 0 and global_state['it'] % print_every == 0:
                logging.info('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f' %
                      (epoch, global_state['it'], train_loss, l))
                save_plot(os.path.join(global_state['args'].out_dir, 'progress.png'))

            if global_state['it'] % save_every == 0:
                is_best = False
                if global_state['best_loss'] > l:
                    global_state['best_loss'] = l
                    global_state['best_it'] = global_state['it']
                    is_best = True
                save_checkpoint(
                    get_checkpoint_file(global_state['args']),
                    model,
                    optimizer,
                    global_state,
                    is_best)
            valid_cnt = 0

            scheduler.step(train_loss)

            #lr_new = lr_scheduler.on_epoch_end(it, lr, train_loss)
            #if lr_new != lr:
            #    lr = lr_new
            #    adjust_learning_rate(scheduler, lr)
            lr_new = get_learning_rate(scheduler.optimizer)
            if lr_new != global_state['lr']:
                logging.info('[%d, %d]\tlearning rate changed from %f to %f' % (epoch, global_state['it'], global_state['lr'], lr_new))
                global_state['lr'] = lr_new
            insert_log(global_state['it'], 'lr', get_learning_rate(scheduler.optimizer))

    return global_state['it'], global_state['best_loss'], global_state['best_it']


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
    img, mask, mask_seg = random_rotate90_transform2(img, mask, mask_seg)
    img = ToTensor()(img)
    mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
    mask_seg = torch.from_numpy(np.transpose(mask_seg, (2, 0, 1))).float()
    return img, mask, mask_seg


def main():
    # global it, best_it, best_loss, LOG, args
    args = parser.parse_args()
    args.override_model_opts = [x.replace('-','_') for x in args.override_model_opts]

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

    global_state = {'it':0,
                    'best_loss':1e20,
                    'best_it':0,
                    'lr':args.lr,
                    'args':args,
                    'grad_accum_it':0,
                    'bp_wt_sum':0.05,
                    'bp_wt_cnt': 10,}

    # create model

    trainer = None
    model = None
    optimizer = None
    scheduler = None
    if args.model == 'knn':
        trainer = train_knn
        model = KNN()

    elif args.model == 'cnn':
        trainer=train_cnn
        model = CNN()
    else:
        raise ValueError("Only supported models are cnn or knn")

    logging.info(model)
    logging.info('number of parameters: %d' % sum([param.nelement() for param in model.parameters()]))
    
    # set up optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                           weight_decay = args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr,
                           momentum = args.momentum,
                           weight_decay = args.weight_decay)
    else:
        raise ValueError('unknown optimization: %s' % args.optim)

    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, cooldown=args.cooldown, min_lr=args.min_lr, verbose=1)


    if args.use_instance_weights > 0 and args.criterion != 'bce':
        raise ValueError('instance weights currently only supported for bce criterion')
    # define loss function (criterion)
    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'bce':
        if args.use_instance_weights > 0:
            criterion = nn.BCEWithLogitsLoss(torch.ones((1)))
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'dice':
        criterion = loss.DiceLoss()
    elif args.criterion == 'jaccard':
        criterion = loss.JaccardLoss()
    else:
        raise ValueError('unknown criterion: %s' % args.criterion)

    # optionally resume from a checkpoint
    if args.resume:
        load_checkpoint(args.resume, model, optimizer, global_state)
        # make sure args here is consistent with possibly updated global_state['args']!
        args = global_state['args']
        args.force_overwrite = 1
    else:
        # prevent accidental overwriting
        ckpt_file = get_checkpoint_file(args)
        if os.path.isfile(ckpt_file) and args.force_overwrite == 0:
            raise ValueError('checkpoint already exists, exiting: %s' % ckpt_file)
        clear_log()

    # Data loading
    logging.info('loading data')

    def load_data():
        return NucleusDataset(args.data, args.stage, transform=train_transform)
    timer = timeit.Timer(load_data)
    t,dset = timer.timeit(number=1)
    logging.info('load time: %.1f' % t)


    # hack: this image format (1388, 1040) occurs only ones, stratify complains ..
    dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

    stratify = None
    if args.stratify > 0:
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
        it, best_loss, best_it = trainer(train_loader, valid_loader, model, criterion, optimizer, scheduler, epoch, args.eval_every, args.print_every, args.save_every, global_state)
        logging.info('final best: it = %d, valid = %.5f' % (best_it, best_loss))


if __name__ == '__main__':
    main()
