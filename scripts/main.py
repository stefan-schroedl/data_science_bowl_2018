#!//usr/bin/env python

import sys
import os
import shutil

import configargparse
import copy
import timeit
import time
import time
import logging
import math
import re
import glob

from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import torch
from torch import optim, nn

from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR,  MultiStepLR
from my_lr_scheduler import ReduceLROnPlateau2
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

from sklearn.neighbors import KNeighborsClassifier

import skimage
from skimage.color import rgb2grey

import utils
from utils import mkdir_p, csv_list, int_list, strip_end, init_logging, get_log, set_log, clear_log, insert_log, get_latest_log, get_history_log, prob_to_rles, labels_to_rles, numpy_to_torch, torch_to_numpy
from adjust_learn_rate import get_learning_rate, adjust_learning_rate

from KNN import *

import nuc_trans
from nuc_trans import as_segmentation, separate_touching_nuclei, erode_mask, redilate_mask, noop_augmentation, nuc_augmentation
import dataset
from dataset import NucleusDataset

import architectures
from architectures import CNN, UNetClassify, init_weights

import post_process
from post_process import parametric_pipeline, parametric_pipeline_v1, parametric_pipeline_orig
import loss
from loss import iou_metric, diagnose_errors, show_compare_gt, union_intersection, precision_at

from meter import AverageMeter


class TrainingBlowupError(Exception):
    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super(TrainingBlowupError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

def get_checkpoint_file(args, it=0):
    if it > 0:
        return os.path.join(args.out_dir, 'model_save_%s.%d.pth.tar' % (args.experiment, it))
    return os.path.join(args.out_dir, 'model_save_%s.pth.tar' % args.experiment)


def get_latest_checkpoint_file(args):
    last_it = -1
    last_ckpt = ''
    pattern = re.compile('model_save_%s.(?P<it>[0-9]+).pth.tar' % args.experiment)
    for path, dirs, files in os.walk(args.out_dir):
        for file in files:
            m = pattern.match(file)
            if m:
                it = int(m.group(1))
                if it > last_it:
                    last_it = it
                    last_ckpt = file
    if last_it == -1:
        raise ValueError('no previous checkpoint')
    return os.path.join(args.out_dir, last_ckpt)


def save_plot(fname, title=None):
    train_loss, train_loss_it = get_history_log('train_loss')
    valid_loss, valid_loss_it = get_history_log('valid_loss')
    epoch_loss, epoch_loss_it = None, None
    try:
        epoch_loss, epoch_loss_it = get_history_log('epoch_train_loss')
    except:
        pass
    grad, grad_it = get_history_log('grad')

    fig, ax = plt.subplots(2, 1)
    if title is not None:
        fig.suptitle(title)
    ax[0].plot(train_loss_it, train_loss, 'g', label='tr')
    if epoch_loss is not None:
        ax[0].plot(epoch_loss_it, epoch_loss, 'b', label='e')
    ax[0].plot(valid_loss_it, valid_loss, 'r', label='ts')
    ax[1].plot(grad_it, grad, label='grad')
    ax[0].grid(True, 'both')
    ax[0].legend()
    ax[1].grid(True, 'both')
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
parser.add('--out-dir', '-o', help='output directory')
parser.add('--stage', '-s', default='stage1',
           help='stage [default: %(default)s]')
parser.add('--group', '-g', default='train',
           help='group name [default: %(default)s]')
#parser.add('--arch', '-a', metavar='ARCH', default='resnet18',
#                                        choices=model_names,
#                                        help='model architecture: ' +
#                                            ' | '.join(model_names) +
#                                            ' (default: resnet18)')
parser.add('-j', '--workers', default=1, type=int, metavar='N',
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
parser.add('--scheduler', default='none', choices=['none', 'plateau', 'exp', 'multistep'],
           help='learn rate scheduler [default: %(default)s]')
parser.add('--scheduler_milestones', type=int_list, default='200', help='milestones for multistep scheduler')
parser.add('--min-lr', default=0.0001, type=float,
           metavar='N', help='minimum learn rate for scheduler [default: %(default)s]')
parser.add('--momentum', '-m', default=0.9, type=float, metavar='M',
           help='momentum [default: %(default)s]')
parser.add('--weight-decay', default=1e-4, type=float,
           metavar='W', help='weight decay [default: %(default)s]')
parser.add('--history-size', type=int, default=100, help='history size for lbfgs [default: %(default)s]')
parser.add('--max-iter-lbfgs', type=int, default=20, help='maximum iterations for lbfgs [default: %(default)s]')
parser.add('--tolerance-change', type=float, default=0.01, help='tolerance for termination for lbfgs [default: %(default)s]')
parser.add('--weight-init', default='kaiming', choices=['kaiming', 'xavier', 'default'],
           help='weight initialization method default: %(default)s]')
parser.add('--use-instance-weights', default=0, type=int,
           metavar='N', help='apply instance weights during training [default: %(default)s]')
parser.add('--clip-gradient', default=0.25, type=float,
           metavar='C', help='clip excessive gradients during training [default: %(default)s]')
parser.add('--criterion', '-C', default='bce', choices=['mse','bce','jaccard','dice'],
           metavar='C', help='loss function [default: %(default)s]')
parser.add('--optim', '-O', default='sgd', choices=['sgd','adam','lbfgs'],
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
parser.add('--patience', default=3, type=int,
           metavar='N', help='patience for lr scheduler, in epochs [default: %(default)s]')
parser.add('--patience-threshold', default=.1, type=float,
           metavar='N', help='patience threshold for lr scheduler [default: %(default)s]')
parser.add('--cooldown', default=5, type=int,
           metavar='N', help='cooldown for lr scheduler [default: %(default)s]')
parser.add('--lr-decay', default=.1, type=float,
           metavar='N', help='decay factor for lr scheduler [default: %(default)s]')
parser.add('--switch-to-lbfgs', default=0, type=int,
           metavar='N', help='if lr scheduler reduces rate, switch to lbfgs [default: %(default)s]')
parser.add('--resume', default='', type=str, metavar='PATH',
           help='path to latest checkpoint [default: %(default)s]')
parser.add('--override-model-opts', type=csv_list, default='override-model-opts,resume,experiment,out-dir,save-every,print-every,eval-every,scheduler,log-file',
           help='when resuming, change these options [default: %(default)s]')
parser.add('--calc-iou', type=int, default=0, help='calculate iou and exit')
parser.add('--calc-pred', type=int, default=0, help='calculate predictions and exit')
parser.add('--predictions-file', type=str, default='predictions.csv', help='file name for predictions output')
parser.add('--random-seed', type=int, default=2018, help='set random number generator seed [default: %(default)s]')
parser.add('--verbose', '-V', type=int, default=0, help='verbose logging')
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
         'model': model,
         'log': get_log()}

    if global_state:
            s['global_state'] = global_state

    if optimizer:
        s['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(s, fname)

    if is_best:
        logging.info('new best: it = %d, train = %.5f, valid = %.5f' % (get_latest_log('it')[0], get_latest_log('train_loss', float('nan'))[0], get_latest_log('valid_loss', float('nan'))[0]))
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

    #if model:
    #    model.load_state_dict(checkpoint['model_state_dict'])

    old_model = None
    if 'model' in checkpoint:
        old_model = checkpoint['model']

    if optimizer and (not global_state['args'] or not('optim' in global_state['args'].override_model_opts)):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if global_state and 'global_state' in checkpoint:
        for k,v in checkpoint['global_state'].iteritems():
            if k != 'args' and k not in global_state['args'].override_model_opts:
                global_state[k] = v

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
        "=> loaded checkpoint '{}' (iteration {})\n".format(
            fname, it))
    return old_model


def validate_knn(model, loader, criterion):
    running_loss = 0.0
    cnt = 0
    for i, (img, (labels, labels_seg)) in tqdm(enumerate(loader), 'test'):
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


def torch_pred_to_np_label(pred, sz=2, max_clusters_for_dilation=100, thresh=0.0):
    if not isinstance(pred, np.ndarray):
        pred_np = pred.data.numpy().squeeze()
    img_th = (pred_np > thresh).astype(int)
    img_l = redilate_mask(img_th, sz=sz, skip_clusters=max_clusters_for_dilation)
    return img_l, pred_np


def validate(model, loader, criterion, calc_iou=False, max_clusters_for_dilation=100):

    time_start = time.time()

    model.eval()
    #model.train() # for some reason, batch norm doesn't work properly with eval mode!!!

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        criterion.weight = torch.ones(1) # reset weight if it was changed!
    running_loss = 0.0
    cnt = 0
    sum_iou = 0.0

    for i, row in tqdm(enumerate(loader), desc='test', total=loader.__len__()):

        img, labels_seg = Variable(row['images'], requires_grad=False), Variable(row['masks_seg'], requires_grad=False)
        pred = model(img)
        pred_lab = pred

        # HACK for upper bound/sanity check
        #pred = row['masks_seg']
        #pred_lab = Variable(scipy.ndimage.label(row['masks_seg'])[0], requires_grad=False)
        #pred_lab = Variable(row['masks_prep'], requires_grad=False)
        # hmmm ... loss can actually become negative if too good??? numerical problem???
        #pred[pred<=0] = -0.693
        #pred[pred>0] = 0.693
        #pred = Variable(pred, requires_grad=False)

        loss = criterion(pred, labels_seg)
        running_loss += loss.data[0]
        iou = 0.0
        cnt += 1
        if calc_iou:
            pred_l, _ = torch_pred_to_np_label(pred, max_clusters_for_dilation=max_clusters_for_dilation)
            iou = iou_metric(row['masks'].numpy().squeeze(), pred_l)
            if 0:
                logging.info('%s\t%f\t%f' % (row['id'], loss.data[0], iou))
                if row['id'][0] == 'bbfc4aab5645637680fa0ef00925eea733b93099f1944c0aea09b78af1d4eef2':
                    fig, ax = plt.subplots(1, 2, figsize=(50, 50))
                    plt.tight_layout()
                    ax[0].imshow(torch_to_numpy(img.data[0]))
                    ax[1].imshow(torch_to_numpy(labels_seg.data))
                    fig.savefig('img_debug_gt.png')
                    plt.close()
                    fig, ax = plt.subplots(1, 2, figsize=(50, 50))
                    ax[0].imshow(torch_to_numpy(pred.data[0]))
                    ax[1].imshow(pred_l)
                    fig.savefig('img_debug.png')
                    plt.close()

            sum_iou += iou
    l = running_loss / cnt
    iou = sum_iou / cnt

    time_end = time.time()
    return l, iou, time_end - time_start


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


def backprop_weight(labels, pred, global_state, thresh=0.1):

    w =  1.0 / (labels.flatten().max() + 1.0)

    if 0:
        #img_th = parametric_pipeline(pred, circle_size=4)
        thresh = 0.5
        img_th = (pred > -0.1).astype(int)
        img_l = scipy.ndimage.label(img_th)[0]
        union, intersection, area_true, area_pred = union_intersection(labels, img_l)

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

    w_norm = w / (global_state['bp_wt_sum'] / global_state['bp_wt_cnt'])

    global_state['bp_wt_sum'] += w
    global_state['bp_wt_cnt'] += 1

    return w_norm


def train_cnn (train_loader,
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

    time_start = time.time()
    time_val = 0.0
    n_val = 0

    is_lbfgs = global_state['args'].optim == 'lbfgs'
    accum_total = global_state['args'].grad_accum

    # slightly ugly: to accommodate  lbfgs with gradient accumulation, we need to control the dataset with low
    # level function

    #for global_state['it'], (img, (labels, labels_seg)) in tqdm(enumerate(train_loader, global_state['it'] + 1)):

    _train_loader = train_loader.__iter__()

    # PYTHON WEIRDNESS: using scalar inside closure gives error! Therefore using arrays with one element
    # https://stackoverflow.com/questions/4851463/python-closure-write-to-variable-in-parent-scope

    acc = [] # train data buffer, needed for gradient accumulation with lbfgs
    inner_cnt = [0]
    running_loss = AverageMeter()
    final_loss = AverageMeter()
    weight_stats = AverageMeter()
    iou_stats = AverageMeter()
    loss_stats = AverageMeter()

    # helper function to do forward and accumulative backward passes on acc buffer
    def closure():
        optimizer.zero_grad()
        logging.debug('start inner %d' % inner_cnt[0])
        loss = 0
        for  row in tqdm(acc, 'train'):
            img, labels_seg = Variable(row['images']), Variable(row['masks_prep_seg'])
            pred = model(img)
            if global_state['args'].use_instance_weights > 0:
                w = backprop_weight(row['masks'].numpy().squeeze(), pred.data[0].numpy().squeeze(), global_state)
                weight_stats.update(w)
                criterion.weight = torch.FloatTensor([w])
                #criterion.weight = row['ov'] * w
                #criterion.weight = pred.data.clone().fill_(w)

            loss = criterion(pred, labels_seg)
            final_loss.update(loss.data[0])
            if inner_cnt[0] == 0: # for lbfgs, only record the first eval!
                running_loss.update(loss.data[0])
                loss_stats.update(loss.data[0])
                pred_l, _ = torch_pred_to_np_label(pred, max_clusters_for_dilation=50) # dilation is slow, skip!
                iou_stats.update(iou_metric(row['masks'].numpy().squeeze(), pred_l))
            logging.debug('loss: %s', loss.data.numpy()[0])

            loss.backward()

        if global_state['args'].clip_gradient > 0:
            gradi = torch.nn.utils.clip_grad_norm(model.parameters(), global_state['args'].clip_gradient)
            if inner_cnt[0] == 0: # for lbfgs, only record the first eval!
                grad[0] = gradi

        inner_cnt[0] += 1

        return loss

    more_data = True
    while more_data:

        acc = []
        inner_cnt = [0]
        running_loss.reset()
        final_loss.reset()
        weight_stats.reset()
        grad = [float('nan')]
        it = global_state['it']

        model.train()
        for i in range(global_state['args'].grad_accum):
            try:
                acc.append(_train_loader.next())
            except StopIteration:
                more_data = False
                break

        num_acc = len(acc)
        if num_acc > 0:

            if not is_lbfgs:
                closure()
                optimizer.step()
            else:
                optimizer.step(closure)

            train_loss = running_loss.avg

            blowup = False
            if math.isnan(train_loss):
                msg = 'iteration %d - training blew up ...' % it
                logging.error(msg)
                raise TrainingBlowupError(msg)

            insert_log(it, 'train_loss', train_loss)

            if not math.isnan(grad[0]):
                insert_log(it, 'grad', grad[0])

            if global_state['args'].use_instance_weights > 0:
                insert_log(it, 'bp_wt', weight_stats.avg)

            if is_lbfgs:
                final_train_loss = final_loss.avg
                logging.debug('final loss: %f', final_train_loss)
                insert_log(it, 'final_train_loss', final_train_loss)

            validated = False
            for i in range(it, it + num_acc):

                if i % eval_every == 0 and not validated:
                    l, iou, t = validate(model, valid_loader, criterion, True)
                    time_val += t
                    n_val += len(valid_loader)
                    insert_log(i, 'valid_loss', l)
                    insert_log(i, 'valid_iou', iou)
                    validated = True

                iou = get_latest_log('valid_iou', 0.0)[0]
                l = get_latest_log('valid_loss', 0.0)[0]

                if i % print_every == 0:
                    logging.info('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f\tiou: %.3f\tlr: %g' %
                                 (epoch, i, train_loss, l, iou, global_state['lr']))
                    if global_state['args'].use_instance_weights > 0:
                        logging.debug('[%d, %d]\tavg instance weight: %.3f' %
                                      (epoch, i, global_state['bp_wt_sum'] /  global_state['bp_wt_cnt']))
                    save_plot(os.path.join(global_state['args'].out_dir, 'progress.png'), global_state['args'].experiment)

                if i % save_every == 0:
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

            global_state['it'] += num_acc

    time_end = time.time()
    time_total = time_end - time_start
    return global_state['it'], global_state['best_loss'], global_state['best_it'], loss_stats.avg, iou_stats.avg, time_total, time_val, n_val


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
        pred = labels_seg.clone()
        pred = torch.clamp(pred, m, m)
        img, labels = Variable(img), Variable(labels_seg)
        pred = Variable(pred)

        loss = criterion(pred, labels_seg)

        running_loss += loss.data[0]
        cnt += 1

    return running_loss/cnt, m


def main():
    # global it, best_it, best_loss, LOG, args
    args = parser.parse_args()

    # in overrides, replace '-' by '_', and check that it is indeed an option
    new_overrides = []
    for opt in args.override_model_opts:
        opt_new = opt.replace('-','_')
        if opt_new not in args.__dict__:
            raise ValueError('overriding option %s does not exist' % opt)
        new_overrides.insert(0, opt_new)
    args.override_model_opts = new_overrides

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
        torch.manual_seed(args.random_seed)

    global_state = {'it':0,
                    'best_loss':1e20,
                    'best_it':0,
                    'lr':args.lr,
                    'args':args,
                    'bp_wt_sum':0.1,
                    'bp_wt_cnt': 10,}

    # create model

    trainer = None
    model = None
    optimizer = None
    if args.model == 'knn':
        trainer = train_knn
        model = KNN()

    elif args.model == 'cnn':
        trainer = train_cnn
        #model = CNN(32)
        model = UNetClassify(layers=4, init_filters=32)
        if args.weight_init != 'default':
           init_weights(model, args.weight_init)
    else:
        raise ValueError("Only supported models are cnn or knn")

    logging.info('model:\n')
    logging.info(model)
    logging.info('number of parameters: %d\n' % sum([param.nelement() for param in model.parameters()]))

    # set up optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                           weight_decay = args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr,
                           momentum = args.momentum,
                           weight_decay = args.weight_decay)
    elif args.optim == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(),
                                lr = args.lr,
                                max_iter = args.max_iter_lbfgs,
                                history_size = args.history_size,
                                tolerance_change= args.tolerance_change)
    else:
        raise ValueError('unknown optimization: %s' % args.optim)

    # set up learn rate scheduler
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau2(optimizer,
                                       factor=args.lr_decay,
                                       patience=args.patience,
                                       patience_threshold=args.patience_threshold,
                                       cooldown=args.cooldown,
                                       min_lr=args.min_lr, verbose=1)
    elif args.scheduler == 'multistep':
        if args.scheduler_milestones is None or len(args.scheduler_milestones) == 0:
            raise ValueError('scheduler-milestones cannot be empty for multi-step')
        scheduler = MultiStepLR(optimizer, args.scheduler_milestones)
    elif args.scheduler == 'exp':
        # dummy for now
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_decay)


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
        model = load_checkpoint(args.resume, model, optimizer, global_state)
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
    def load_data():
        return NucleusDataset(args.data, stage_name=args.stage, group_name=args.group, preprocess=erode_mask, transform=nuc_augmentation)
    timer = timeit.Timer(load_data)
    t,dset = timer.timeit(number=1)
    logging.info('load time: %.1f\n' % t)

    # hack: this image format (1388, 1040) occurs only ones, stratify complains ..
    dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

    stratify = None
    if args.stratify > 0:
        stratify = dset.data_df['images'].map(lambda x: '{}'.format(x.size))
    train_dset, valid_dset = dset.train_test_split(test_size=args.valid_fraction, random_state=args.random_seed, shuffle=True, stratify=stratify, test_transform=noop_augmentation)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=1, shuffle=True)
    if args.calc_iou > 0 or args.calc_pred > 0:
        train_loader.transform = noop_augmentation()

    if args.calc_iou > 0:
        loss, iou, _ = validate(model, train_loader, criterion, calc_iou=True, max_clusters_for_dilation=1e20)
        msg = 'train: loss = %f, iou = %f' % (loss, iou)
        logging.info(msg)
        print msg
        loss, iou, _ = validate(model, valid_loader, criterion, calc_iou=True, max_clusters_for_dilation=1e20)
        msg = 'test: loss = %f, iou = %f' % (loss, iou)
        logging.info(msg)
        print msg
        return

    if args.calc_pred > 0:
        # calculate predictions
        model.eval()
        #model.train() # for some reason, batch norm doesn't work properly with eval mode!!!
        thresh = 0.0
        preds = []
        for i in tqdm(range(len(dset.data_df))):
            img = dset.data_df['images'].iloc[i]
            pred = model(Variable(numpy_to_torch(img, True), requires_grad=False))
            pred_l, pred = torch_pred_to_np_label(pred, max_clusters_for_dilation=1e20) # highest precision
            preds.append(pred_l)

            if 1:
                fig, ax = plt.subplots(1, 3, figsize=(50, 50))
                plt.tight_layout()
                ax[0].imshow(img)
                ax[1].imshow(pred)
                ax[2].imshow(pred_l)
                fig.savefig('img_test.%d.png' % i)
                plt.close()

        dset.data_df['pred'] = preds

        dset.data_df['rles'] = dset.data_df['pred'].map(lambda x: list(labels_to_rles(x)))

        out_pred_list = []
        for _, c_row in tqdm(dset.data_df.iterrows()):
            for c_rle in c_row['rles']:
                out_pred_list.append({'ImageId': c_row['id'],
                                     'EncodedPixels': ' '.join(np.array(c_rle).astype(str))})

        out_pred_df = pd.DataFrame(out_pred_list)
        logging.info('%d regions found for %d images' %(out_pred_df.shape[0], dset.data_df.shape[0]))
        out_pred_df[['ImageId', 'EncodedPixels']].to_csv(args.predictions_file, index = False)

        return

    #l, iou = validate(model, valid_loader, criterion, True)
    #logging.info('initial validation: %.3f %.3f\n' % (l, iou))
    #global_state['best_loss'] = l
    #global_state['best_it'] = 0

    if args.resume is None:
        insert_log(0, 'valid_loss', l)

    logging.info('parameters:\n')
    for k in global_state['args'].__dict__:
        logging.info('> %s = %s' % (k, str(global_state['args'].__dict__[k])))
    logging.info('')
    logging.info('train set: %d; test set: %d' % (len(train_dset), len(valid_dset)))

    recovered_ckpt = None
    recovery_attempts = 0


    for epoch in range(args.epochs):
        try:
            it, best_loss, best_it, epoch_loss, epoch_iou, time_total, time_val, n_val = trainer(train_loader, valid_loader, model, criterion, optimizer, scheduler, epoch, args.eval_every, args.print_every, args.save_every, global_state)

            logging.info('[%d, %d]\tepoch: train loss %.3f, iou=%.3f, total time=%d, val time=%d, it/s=%.2f, train it/s=%.2f, valid it/s=%.2f' %
                         (epoch,
                          global_state['it'],
                          epoch_loss,
                          epoch_iou,
                          time_total,
                          time_val,
                          1.0 * time_total / len(train_loader),
                          1.0 * (time_total - time_val) / len(train_loader),
                          1.0 * time_val / n_val if n_val > 0 else 0.0))


            # check for blowup
            last_epoch_loss = get_latest_log('epoch_train_loss', 1e20)[0]
            if not math.isnan(last_epoch_loss) and epoch_loss > 100.0 * last_epoch_loss:
                msg = 'iteration %d - training blew up ...' % it
                logging.error(msg)
                raise TrainingBlowupError(msg)

            save_checkpoint(
                get_checkpoint_file(global_state['args'], global_state['it']),
                model,
                optimizer,
                global_state)

            insert_log(global_state['it'], 'epoch_train_loss', epoch_loss)
            insert_log(global_state['it'], 'train_iou', epoch_iou)
            insert_log(global_state['it'], 'lr', global_state['lr'])

            if global_state['args'].scheduler != 'none':
                # note: different interface!
                # ReduceLROnPlateau.step() takes metrics as argument,
                # other schedulers take epoch number
                if isinstance(scheduler, ReduceLROnPlateau2):
                    scheduler.step(epoch_loss, epoch)
                else:
                    scheduler.step(epoch)

                lr_new = get_learning_rate(optimizer)
                lr_old = global_state['lr']
                if lr_old != lr_new:
                    if not args.switch_to_lbfgs or isinstance(optimizer, optim.LBFGS):
                        logging.info('[%d, %d]\tLR changed from %f to %f.' %
                                     (epoch, global_state['it'], lr_old, lr_new))
                        global_state['lr'] = lr_new
                    else:
                        logging.info('[%d, %d]\tswitching to lbfgs' %
                                     (epoch, global_state['it']))
                        lr = 0.8
                        optimizer = optim.LBFGS(model.parameters(),
                                                lr = lr,
                                                max_iter = args.max_iter_lbfgs,
                                                history_size = args.history_size,
                                                tolerance_change = args.tolerance_change)
                        global_state['args'].grad_accum = len(train_loader)
                        args.grad_accum = len(train_loader)
                        global_state['args'].optim == 'lbfgs'
                        args.optim = 'lbfgs'
                        global_state['args'].clip_gradient = 1e20
                        args.clip_gradient = 0
                        global_state['args'].scheduler = 'plateau'
                        args.scheduler = 'plateau'
                        global_state['lr'] = lr

                        scheduler = ReduceLROnPlateau2(optimizer,
                                                       factor=0.9,
                                                       patience=1,
                                                       patience_threshold=0.1,
                                                       min_lr=0.1, verbose=1)

            logging.info('epoch best: it = %d, valid = %.5f' % (best_it, best_loss))

        except TrainingBlowupError:
            # numerical instability, try to recover
            # sometime lbfgs blows up, gradient clipping is not applicable
            # restart with reduced learn rate
            if isinstance(optimizer, optim.LBFGS):
                ckpt = get_latest_checkpoint_file(args)
                if recovered_ckpt == ckpt:
                    # have tried previously the same checkpoint, reduce lr
                    recovery_attempts += 1
                else:
                    recovery_attempts = 1
                recovered_ckpt = ckpt

                min_lr = 0.1
                lr = max(global_state['lr'] * .5, min_lr)
                if lr >= global_state['lr']:
                    msg = 'attempt %d: using lbfgs and lr (%f) already at min_lr (%f), giving up' % (recovery_attempts, global_state['lr'], min_lr)
                    logging.error(msg)
                    raise
                model = load_checkpoint(ckpt,
                                model,
                                optimizer,
                                global_state)
                global_state['lr'] = lr
                optimizer = optim.LBFGS(model.parameters(),
                                        lr = lr,
                                        max_iter = args.max_iter_lbfgs,
                                        history_size = args.history_size,
                                        tolerance_change = args.tolerance_change)
                global_state['args'].grad_accum = len(train_loader)
                args.grad_accum = len(train_loader)
                global_state['args'].optim == 'lbfgs'
                args.optim = 'lbfgs'
                global_state['args'].clip_gradient = 1e20
                args.clip_gradient = 0
                global_state['args'].scheduler = 'plateau'
                args.scheduler = 'plateau'

                scheduler = ReduceLROnPlateau2(optimizer,
                                               factor=0.9,
                                               patience=1,
                                               patience_threshold=0.1,
                                               min_lr=0.1, verbose=1)

                logging.error('recovered from checkpoint %s (attempt %d), lr = %f. keeping fingers crossed ...' % (ckpt, recovery_attempts, lr))
            else:
                logging.error('cannot recover ... terminating.')
                raise


if __name__ == '__main__':
    main()
