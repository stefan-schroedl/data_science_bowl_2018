#!//usr/bin/env python

import sys
import os
import shutil

import configargparse
import copy
import timeit
import time
import logging
import math
import re
from glob import glob

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
from utils import mkdir_p, csv_list, int_list, strip_end, init_logging, get_log, set_log, clear_log, insert_log, get_latest_log, get_history_log, prob_to_rles, labels_to_rles, numpy_to_torch, torch_to_numpy, get_latest_checkpoint_file, get_checkpoint_file, checkpoint_file_from_dir, moving_average
from adjust_learn_rate import get_learning_rate, adjust_learning_rate

from KNN import *

import nuc_trans

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


def save_plot(fname, title=None):
    train_loss, train_loss_it = get_history_log('running_train_loss')
    valid_loss, valid_loss_it = get_history_log('running_valid_loss')
    epoch_loss, epoch_loss_it = None, None
    try:
        epoch_loss, epoch_loss_it = get_history_log('epoch_train_loss')
    except:
        pass
    grad, grad_it = get_history_log('running_grad')

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
parser.add('--data', '-d', metavar='DIR', required=True,
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
parser.add('-b', '--batch-size', default=1, type=int,
           metavar='N', help='mini-batch size [default: %(default)s]')
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
parser.add('--cuda', type=int, default=0, help='use cuda')
parser.add('--cuda-benchmark', type=int, default=0, help='use cuda benchmark mode')

# loading and saving checkpoints
# notes:
# 1) pytorch documentation recommends saving and loading only the state of the model
#    (load_state_dict()), but then you would have to track different architectures
#    outside - simplifying for now.
# 2) pytorch recommends instantiating the optimizer *after* the model. What to do if
#    we read the model from file? especially if the model is saved in cuda mode, the
#    connection will be broken. I think for now, we shouldn't reastore the optimizer
#    at all. of course, the result is then not identical if we had continued, since the
#    state is reinitialized (especially for adam).
# 3) Similarly, scheduler state is not saved/restored.


def save_checkpoint(fname,
                    model,
                    optimizer=None,
                    global_state=None,
                    is_best=False):

    #model.clearState()
    s = {'model_state_dict': model.state_dict(),
         'model': model,
         'log': get_log()}

    if global_state:
            s['global_state'] = global_state

    if optimizer:
        s['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(s, fname)

    if is_best:
        pref = strip_end(fname, '.pth.tar')
        shutil.copyfile(fname, '%s_best.pth.tar' % pref)


def load_checkpoint(fname,
                    model,
                    optimizer=None,
                    global_state=None):

    if not os.path.isfile(fname):
        raise ValueError('checkpoint not found: %s', fname)

    checkpoint = torch.load(fname, map_location='cpu') # always load to cpu first!
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
        pred_np = pred.data.cpu().numpy().squeeze()
    img_th = (pred_np > thresh).astype(int)
    img_l = nuc_trans.redilate_mask(img_th, sz=sz, skip_clusters=max_clusters_for_dilation)
    return img_l, pred_np


def validate(model, loader, criterion, calc_iou=False, max_clusters_for_dilation=100, calc_baseline=False):
    # calc_baseline=True -> calculate loss when predicting constant global average

    time_start = time.time()

    model.eval()
    #model.train() # for some reason, batch norm doesn't work properly with eval mode!!!

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        criterion.weight = dev(torch.ones(1)) # reset weight if it was changed!

    avg_loss = AverageMeter()
    avg_iou = AverageMeter()
    if not calc_iou:
        avg_iou.update(0.0)

    avg_mask = AverageMeter()
    if calc_baseline:
        for i, row in enumerate(loader):
            avg_mask.update(row['masks_bin'].numpy().mean())

    for i, row in tqdm(enumerate(loader), desc='valid', total=loader.__len__()):

        img, labels_bin = Variable(dev(row['images_prep']), volatile=True), Variable(dev(row['masks_bin']), volatile=True)

        if not calc_baseline:
            pred = model(img)
            pred_lab = pred
        else:
            pred = dev(torch.ones_like(labels_bin) * avg_mask.avg)

        # HACK for upper bound/sanity check
        #pred = row['masks_bin']
        #pred_lab = Variable(scipy.ndimage.label(row['masks_bin'])[0], volatile=True)
        #pred_lab = Variable(row['masks_prep'], volatile=True)
        # hmmm ... loss can actually become negative if too good??? numerical problem???
        #pred[pred<=0] = -0.693
        #pred[pred>0] = 0.693
        #pred = Variable(pred, volatile=True)

        loss = criterion(pred, labels_bin)
        avg_loss.update(loss.data[0])

        if calc_iou:
            pred_l, _ = torch_pred_to_np_label(pred, max_clusters_for_dilation=max_clusters_for_dilation)
            iou = iou_metric(row['masks'].numpy().squeeze(), pred_l)
            avg_iou.update(iou)
            if 0:
                logging.info('%s\t%f\t%f' % (row['id'], loss.data[0], iou))
                if row['id'][0] == 'bbfc4aab5645637680fa0ef00925eea733b93099f1944c0aea09b78af1d4eef2':
                    fig, ax = plt.subplots(1, 2, figsize=(50, 50))
                    plt.tight_layout()
                    ax[0].imshow(torch_to_numpy(img.data[0]))
                    ax[1].imshow(torch_to_numpy(labels_bin.data))
                    fig.savefig('img_debug_gt.png')
                    plt.close()
                    fig, ax = plt.subplots(1, 2, figsize=(50, 50))
                    ax[0].imshow(torch_to_numpy(pred.data[0]))
                    ax[1].imshow(pred_l)
                    fig.savefig('img_debug.png')
                    plt.close()

    time_end = time.time()
    return avg_loss.avg, avg_iou.avg, time_end - time_start


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
    for global_state['it'], (img, (labels, labels_bin)) in tqdm(enumerate(train_loader, global_state['it'] + 1)):
        model.prepare_fit(img,labels,labels_bin)

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

    # lbfgs has to be called with a closure, in contrast to other optimizers

    # PYTHON WEIRDNESS: using scalar inside closure gives error! Therefore using arrays with one element
    # https://stackoverflow.com/questions/4851463/python-closure-write-to-variable-in-parent-scope

    acc = [] # train data buffer, needed for gradient accumulation with lbfgs
    running_loss = AverageMeter() # training loss before gradient step
    epoch_loss = AverageMeter()
    weight_stats = AverageMeter()
    epoch_iou = AverageMeter()

    # only meaningful for lbfgs, loss before last descent in closure
    epoch_loss_last_closure = AverageMeter()
    running_loss_last_closure = AverageMeter()
    closure_cnt = [0] # (only) for lbfgs, closure can be called multiple times

    # helper function to do forward and accumulative backward passes on acc buffer

    def closure():
        optimizer.zero_grad()
        logging.debug('start closure %d' % closure_cnt[0])
        loss = 0
        running_loss_last_closure.reset()
        for  mb_acc in acc:
            img, labels_bin = Variable(dev(mb_acc['images_prep']), requires_grad=False), Variable(dev(mb_acc['masks_prep_bin']), requires_grad=False)
            pred = model(img)
            if global_state['args'].use_instance_weights > 0:
                w = mb_acc['num_nuc_inv']
                for i in range(w.size()[0]):
                    weight_stats.update(w[i])
                w = w.float().unsqueeze(1).unsqueeze(1).unsqueeze(1) # make size broadcastable with batch dimension
                criterion.weight = dev(w)

            loss = criterion(pred, labels_bin)
            running_loss_last_closure.update(loss.data[0])
            if closure_cnt[0] == 0: # for lbfgs, only record the first eval!
                running_loss.update(loss.data[0])
                epoch_loss.update(loss.data[0])
                for n in range(pred.size()[0]):
                    # WARNING: iou for training is just an approximation, would need 'img' and 'masks' instead of 'img_prep and 'masks_prep' (original size)
                    pred_l, _ = torch_pred_to_np_label(pred[n], max_clusters_for_dilation=50) # dilation is slow, skip!
                    epoch_iou.update(iou_metric(mb_acc['masks_prep'][n].numpy().squeeze(), pred_l))
            logging.debug('loss: %s', loss.data.cpu().numpy()[0])
            loss.backward()

        if global_state['args'].clip_gradient > 0:
            gradi = torch.nn.utils.clip_grad_norm(model.parameters(), global_state['args'].clip_gradient)
            if closure_cnt[0] == 0: # for lbfgs, only record the first eval!
                grad[0] = gradi

        closure_cnt[0] += 1

        epoch_loss_last_closure.update(running_loss_last_closure)
        return loss

    acc = []

    total_batches = len(train_loader)
    it_start = global_state['it']
    it_last = it_start + total_batches
    for global_state['it'], mb in tqdm(enumerate(train_loader, global_state['it'] + 1), desc='train', total=total_batches):

        it = global_state['it']

        acc.append(mb)
        if len(acc) < global_state['args'].grad_accum and it < it_last: # last grad accum can be incomplete
            continue

        closure_cnt = [0]
        running_loss.reset()
        running_loss_last_closure.reset()
        weight_stats.reset()
        grad = [float('nan')]

        # learn!
        model.train()
        if not is_lbfgs:
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)

        num_acc = len(acc)
        acc = []
        train_loss = running_loss.avg

        blowup = False
        if math.isnan(train_loss):
            msg = 'iteration %d - training blew up ...' % it
            logging.error(msg)
            raise TrainingBlowupError(msg)

        validated = False # when doing grad accum, don't print validation results twice
        for i in range(it - num_acc + 1, it + 1):
            if i % eval_every == 0 and not validated:
                l, iou, t = validate(model, valid_loader, criterion, True)
                time_val += t
                n_val += len(valid_loader)
                insert_log(i, 'running_valid_loss', l)
                insert_log(i, 'running_valid_iou', iou)
                validated = True

            iou = get_latest_log('running_valid_iou', float('nan'))[0]
            l = get_latest_log('running_valid_loss', float('nan'))[0]

            if i % print_every == 0:
                logging.info('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f\tiou: %.3f\tlr: %g' %
                             (epoch, i, train_loss, l, iou, global_state['lr']))
                if global_state['args'].use_instance_weights > 0:
                    logging.debug('[%d, %d]\tavg instance weight: %.3f' %
                                  (epoch, i, global_state['bp_wt_sum'] /  global_state['bp_wt_cnt']))
                save_plot(os.path.join(global_state['args'].out_dir, 'progress.png'), global_state['args'].experiment)

            if i % save_every == 0:
                is_best = False
                #if global_state['best_loss'] > l:
                #    global_state['best_loss'] = l
                #    global_state['best_it'] = global_state['it']
                # smooth values over iterations
                if iou > 0.0:
                    h = get_history_log('running_valid_iou')
                    n = min(5, len(h[0]))
                    m = moving_average(h[0], n)
                    cur = m[-1]
                    if global_state['best_iou'] < cur:
                        global_state['best_iou'] = cur
                        global_state['best_iou_it'] = global_state['it']
                        is_best = True
                        logging.info('new best: it = %d, loss = %.5f, iou = %.5f' % (global_state['it'], l, cur))

                save_checkpoint(
                    get_checkpoint_file(global_state['args']),
                    model,
                    optimizer,
                    global_state,
                    is_best)

        insert_log(it, 'running_train_loss', train_loss)

        if not math.isnan(grad[0]):
            insert_log(it, 'running_grad', grad[0])

        if global_state['args'].use_instance_weights > 0:
            insert_log(it, 'running_wt', weight_stats.avg)

        if is_lbfgs:
            final_train_loss = running_loss_last_closure.avg
            logging.debug('initial loss: %.3f, final loss: %.3f' %(train_loss, final_train_loss))
            insert_log(it, 'running_loss_last_closure', final_train_loss)

    time_end = time.time()
    time_total = time_end - time_start
    return global_state['it'], epoch_loss.avg, epoch_iou.avg, epoch_loss_last_closure.avg, time_total, time_val, n_val


# choose cpu or gpu
def dev(x):
    return x

def main():
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

    # for later info, save the current configuration and source files
    if args.config is not None and os.path.isfile(args.config):
        shutil.copy(args.config, args.out_dir)

    for f in glob('*.py'):
        shutil.copy(f, args.out_dir)

    if args.log_file is None:
        args.log_file = os.path.join(args.out_dir, '%s.log' % args.experiment)

    init_logging(args)

    if args.cuda > 0:
        if not torch.cuda.is_available():
            raise ValueError('cuda requested, but not available')

        # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
        # note: actually makes it a lot slower on this problem!
        torch.backends.cudnn.benchmark = (args.cuda_benchmark > 0)
        torch.backends.cudnn.enabled   = True
        global dev
        dev = lambda x: x.cuda()

        print '\tset cuda environment'
        print '\t\ttorch.__version__              =', torch.__version__
        print '\t\ttorch.version.cuda             =', torch.version.cuda
        print '\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version()
        try:
            NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
            print '\t\tos[\'CUDA_VISIBLE_DEVICES\']  =', os.environ['CUDA_VISIBLE_DEVICES']
        except Exception:
            print '\t\tos[\'CUDA_VISIBLE_DEVICES\']  =','None'
            NUM_CUDA_DEVICES = 1

        print '\t\ttorch.cuda.device_count()   =', torch.cuda.device_count()
        print '\t\ttorch.cuda.current_device() =', torch.cuda.current_device()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    global_state = {'it':0,
                    'best_loss':1e20,
                    'best_it':0,
                    'best_iou':0.0,
                    'best_iou_it':0,
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
        model = UNetClassify(layers=4, init_filters=16)
        if args.weight_init != 'default':
           init_weights(model, args.weight_init)
        model = dev(model)
    else:
        raise ValueError("Only supported models are cnn or knn")

    # optionally resume from a checkpoint
    if args.resume:
        model = load_checkpoint(checkpoint_file_from_dir(args.resume), model, None, global_state)
        # make sure args here is consistent with possibly updated global_state['args']!
        args = global_state['args']
        args.force_overwrite = 1
        model = dev(model)
    else:
        # prevent accidental overwriting
        ckpt_file = get_checkpoint_file(args)
        if os.path.isfile(ckpt_file) and args.force_overwrite == 0:
            raise ValueError('checkpoint already exists, exiting: %s' % ckpt_file)
        clear_log()

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

    # create criterion
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

    criterion = dev(criterion)

    # Data loading
    def load_data():
        return NucleusDataset(args.data, stage_name=args.stage, group_name=args.group, dset_type = 'test' if args.calc_pred > 0 else 'train')
    timer = timeit.Timer(load_data)
    t,dset = timer.timeit(number=1)
    logging.info('load time: %.1f\n' % t)

    stratify = None
    if args.stratify > 0:
        # hack: this image format (1388, 1040) occurs only once, stratify complains ..
        dset.data_df = dset.data_df[dset.data_df['size'] != (1388, 1040)]

        stratify = dset.data_df['images'].map(lambda x: '{}'.format(x.size))

    if args.calc_pred > 0:
        dset.preprocess()
        # calculate predictions
        model.eval()
        #model.train() # for some reason, batch norm doesn't work properly with eval mode!!!
        preds = []
        for i in tqdm(range(len(dset.data_df))):
            img = dset.data_df['images_prep'].iloc[i]
            pred = model(Variable(dev(numpy_to_torch(img, True)), volatile=True))
            pred_l, pred = torch_pred_to_np_label(pred, max_clusters_for_dilation=1e20) # highest precision
            preds.append(pred_l)

            if 0:
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

    # split data
    calc_baseline = False
    batch_size_train = args.batch_size
    if args.calc_iou > 0 or args.calc_pred > 0 or calc_baseline:
        # originals have uneven dimensions!
        batch_size_train = 1
    batch_size_valid = 1
    train_dset, valid_dset = dset.train_test_split(test_size=args.valid_fraction, random_state=args.random_seed, shuffle=True, stratify=stratify)
    train_loader = DataLoader(train_dset, batch_size=batch_size_train, shuffle=True, pin_memory=(args.cuda > 0), num_workers=args.workers)
    valid_loader = DataLoader(valid_dset, batch_size=batch_size_valid, shuffle=True, pin_memory=(args.cuda > 0), num_workers=args.workers)

    if args.calc_iou > 0 or calc_baseline:
        train_dset.dset_type = 'valid'

    train_dset.preprocess()
    valid_dset.preprocess()

    if args.calc_iou > 0:
        loss, iou, _ = validate(model, train_loader, criterion, calc_iou=True, max_clusters_for_dilation=1e20)
        msg = 'train: loss = %f, iou = %f' % (loss, iou)
        logging.info(msg)
        print msg
        loss, iou, _ = validate(model, valid_loader, criterion, calc_iou=True, max_clusters_for_dilation=1e20)
        msg = 'valid: loss = %f, iou = %f' % (loss, iou)
        logging.info(msg)
        print msg
        return

    if calc_baseline:
        loss, _, _ = validate(model, train_loader, criterion, calc_iou=False, calc_baseline=True)
        msg = 'train: loss = %f' % loss
        logging.info(msg)
        print msg
        loss, _, _ = validate(model, valid_loader, criterion, calc_iou=False, calc_baseline=True)
        msg = 'valid: loss = %f' % loss
        logging.info(msg)
        print msg
        return

    #l, iou = validate(model, valid_loader, criterion, True)
    #logging.info('initial validation: %.3f %.3f\n' % (l, iou))
    #global_state['best_loss'] = l
    #global_state['best_it'] = 0

    if args.resume is None:
        insert_log(0, 'running_valid_loss', l)

    logging.info('command line options:\n')
    for k in global_state['args'].__dict__:
        logging.info('> %s = %s' % (k, str(global_state['args'].__dict__[k])))
    logging.info('')
    logging.info('train set: %d; test set: %d' % (len(train_dset), len(valid_dset)))

    recovered_ckpt = None
    recovery_attempts = 0


    for epoch in range(args.epochs):
        try:
            it, epoch_loss, epoch_iou, epoch_final_loss, time_total, time_val, n_val = trainer(train_loader, valid_loader, model, criterion, optimizer, scheduler, epoch, args.eval_every, args.print_every, args.save_every, global_state)

            logging.info('[%d, %d]\tepoch: train loss %.3f, iou=%.3f, final loss=%.3f, total time=%d, val time=%d, s/ex=%.2f, train s/ex=%.2f, valid s/ex=%.2f' %
                         (epoch,
                          global_state['it'],
                          epoch_loss,
                          epoch_iou,
                          epoch_final_loss,
                          time_total,
                          time_val,
                          1.0 * time_total / len(train_dset),
                          1.0 * (time_total - time_val) / len(train_dset),
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
            insert_log(global_state['it'], 'epoch_final_loss', epoch_final_loss)
            insert_log(global_state['it'], 'epoch_train_iou', epoch_iou)
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
