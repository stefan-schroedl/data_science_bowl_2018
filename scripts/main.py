#!/usr/bin/env python

import sys
import os
import shutil

import configargparse
import copy
import timeit
import time
import logging
import math
from glob import glob
from collections import OrderedDict
import subprocess

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import torch
from torch import optim, nn

from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from reduce_lr_on_plateau2 import ReduceLROnPlateau2
from torch.utils.data import DataLoader

import cv2

from meter import Meter, NamedMeter

from img_proc import numpy_img_to_torch, torch_img_to_numpy, postprocess_prediction
from utils import mkdir_p, csv_list, int_list, float_dict, strip_end, init_logging, get_the_log, set_the_log, clear_log, list_log_keys, insert_log, get_latest_log, get_log, labels_to_rles, get_latest_checkpoint_file, get_checkpoint_file, checkpoint_file_from_dir, moving_average, as_py_scalar, stop_current_instance, get_learning_rate

from KNN import *

from dataset import NucleusDataset

from architectures import CNN, UNetClassify, UNetClassifyMulti, init_weights

import loss
from loss import iou_metric, union_intersection, precision_at


class TrainingBlowupError(Exception):
    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super(TrainingBlowupError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors


def save_plot(fname, title=None):
    train_loss, train_loss_it = get_log('train_last_loss')
    valid_loss, valid_loss_it = get_log('valid_avg_loss')
    epoch_loss, epoch_loss_it = None, None
    try:
        epoch_loss, epoch_loss_it = get_log('train_avg_loss')
    except BaseException:
        pass
    grad, grad_it = get_log('train_avg_grad')

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

    # model.clearState()
    s = {'model_state_dict': model.state_dict(),
         'model': model,
         'log': get_the_log()}

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

    # always load to cpu first!
    checkpoint = torch.load(fname, map_location='cpu')
    try:
        set_the_log(checkpoint['log'])
    except BaseException:
        pass

    # if model:
    #    model.load_state_dict(checkpoint['model_state_dict'])

    old_model = None
    if 'model' in checkpoint:
        old_model = checkpoint['model']

    if optimizer and (not global_state['args'] or not(
            'optim' in global_state['args'].override_model_opts)):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if global_state and 'global_state' in checkpoint:
        for k, v in checkpoint['global_state'].iteritems():
            if k != 'args' and k not in global_state['args'].override_model_opts:
                global_state[k] = v

    if global_state and 'args' in global_state and 'global_state' in checkpoint and 'args' in checkpoint[
            'global_state']:
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
                        logging.warn(
                            ' overriding option %s, old = %s, new = %s' %
                            (k, v_old, v_new))
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
        p_img, p_seg, p_boundary, p_blend = model.predict(img)
        torch_p_seg = torch.from_numpy(
            p_seg[None, :, :].astype(np.float) / 255).float()
        #torch_p_boundary = torch.from_numpy(p_boundary[None,:,:].astype(np.float)/255).float()
        #torch_p_blend = torch.from_numpy(p_blend[None,:,:].astype(np.float)/255).float()
        border = np.full((p_img.shape[0], 5, 3), 255).astype(np.uint8)
        cv2.imshow('img and reconstructed img', np.concatenate(
            (torch_img_to_numpy(img), border, p_img), axis=1))
        cv2.imshow('seg and reconstructed seg', np.concatenate(
            (torch_img_to_numpy(labels_seg), border[:, :, :1], p_seg[:, :, None]), axis=1))
        # cv2.imshow('pbound',p_boundary)
        # cv2.imshow('pblend',p_blend)
        cv2.waitKey(10)
        loss = criterion(Variable(torch_p_seg), Variable(labels_seg.float()))
        running_loss += loss.data[0]
        cnt = cnt + 1
    l = running_loss / cnt
    return l


def dev(x):
    """transparently choose cpu or gpu"""
    return x


def run_model(model, input, train=True):
    if train:
        model.train()
        input = Variable(dev(input), requires_grad=False)
    else:
        model.eval()
        input = Variable(dev(input), volatile=True)
    return model(input)


def apply_weights(data_row, targets, instance_weight_field, use_class_weights, meter):

    """set instance and class weights for targets"""

    if instance_weight_field is None and not use_class_weights:
        w = torch.ones(1)
    else:
        if instance_weight_field is None:
            batch_size = data_row[targets[0]['col']].size()[0]
            w = torch.ones(batch_size, 1)
        else:
            w = data_row[instance_weight_field]

            for i in range(w.size()[0]):
                meter.update(instance_weight_field, as_py_scalar(w[i]))

        # make size broadcastable with batch dimension
        w = w.view(-1, 1, 1, 1)

    w = dev(w)

    # apply class weight

    for target in targets:
        if not use_class_weights or target['w_class'] == 1.0:
            target['crit'].weight = w
        else:
            t = data_row[target['col']]
            sz = t.size()
            w_class = w.clone().repeat(1, sz[1], sz[2], sz[3])
            w_class[t > 0.0] *= target['w_class']
            target['crit'].weight = w_class


def apply_criteria(data_row,
                   pred,
                   targets,
                   meter,
                   instance_weight_field=None,
                   use_class_weights=False,
                   calc_iou=False,
                   pred_field_iou='seg',
                   target_field_iou='masks_prep',
                   max_clusters_for_dilation=100):

    """for one row of input, run the model, evaluate the criteria, and update stats"""

    # if necessary, replicate predictions in case of single-target model

    if not isinstance(pred, (dict, OrderedDict)):
        if len(targets) > 1:
            logging.warn('apparently using single-task model for multiple tasks')

        pred = dict([(target['name'],pred) for target in targets])


    # set weights in criteria

    apply_weights(data_row, targets, instance_weight_field, use_class_weights, meter)


    # apply criteria

    losses = {}
    for i, target_spec in enumerate(targets):

        name = target_spec['name']
        criterion = target_spec['crit']
        target = Variable(dev(data_row[target_spec['col']]), requires_grad=False)

        if not name in pred:
            raise ValueError('model did not predict configured target "%s"' % name)

        l = criterion(pred[name], target)
        losses[name] = l
        meter.update(name, as_py_scalar(l))


    # calculate iou

    if calc_iou:
        pred_seg = pred[pred_field_iou]
        for n in range(pred_seg.size()[0]):
            pred_l, _ = postprocess_prediction(pred_seg[n], max_clusters_for_dilation=max_clusters_for_dilation)
            meter.update('iou',
                iou_metric(data_row[target_field_iou][n].numpy().squeeze(), pred_l))


    # sum total loss, and add it to meter

    total_loss = torch.sum(torch.cat([losses[target['name']] * target['w_crit'] for target in targets]))
    meter.update('loss', as_py_scalar(total_loss))
    return total_loss


def transfer_data(row, targets, input_field, instance_weight_field=None):
    """
    transfer images and masks to gpu.

    hopefully the transfer can be parallelized with model execution.
"""

    fields = [input_field]
    fields.extend([t['col'] for t in targets])
    if instance_weight_field is not None:
        fields.append(instance_weight_field)
    for field in fields:
        row[field] = dev(row[field])


def validate(
        stats,
        loader,
        targets,
        model,
        input_field,
        instance_weight_field=None,
        use_class_weights=False,
        calc_iou=False,
        max_clusters_for_dilation=100,
        calc_baseline=False,
        desc='valid'):
    # calc_baseline=True -> calculate loss when predicting constant global
    # average

    time_start = time.time()

    model.eval()

    avg_mask = NamedMeter()

    if calc_baseline:
        for i, row in enumerate(loader):
            for target in targets:
                avg_mask.update(target['col'], row[target['col']].numpy().mean())
        for target in targets:
            msg = 'avg for column "%s": %.3g' % (target['col'], avg_mask[target['col']].avg)
            print msg
            logging.info(msg)


    for i, row in tqdm(enumerate(loader), desc=desc, total=loader.__len__()):

        transfer_data(row, targets, input_field, instance_weight_field)

        if calc_baseline:
            pred = {}
            for target in targets:
                # constant mean prediction in the same shape as the target
                pred[target['name']] = Variable(dev(torch.ones_like(row[target['col']]) * avg_mask[target['col']].avg),
                                                     volatile=True)
        else:
            pred = run_model(model, row[input_field], train=False)

        total_loss = apply_criteria(row,
                                    pred,
                                    targets,
                                    stats,
                                    instance_weight_field=instance_weight_field,
                                    use_class_weights=use_class_weights,
                                    calc_iou=calc_iou,
                                    pred_field_iou='seg',
                                    target_field_iou='masks_prep',
                                    max_clusters_for_dilation=max_clusters_for_dilation)

    time_end = time.time()
    stats.update('time', time_end - time_start)


def make_submission(dset, model, args, pred_field_iou='seg'):

    dset.preprocess()
    model.eval()

    preds = []
    for i in tqdm(range(len(dset.data_df))):
        img = dset.data_df[args.input_field].iloc[i]
        pred = model(
            Variable(dev(numpy_img_to_torch(img, True)), volatile=True))

        pred_l, pred_seg = postprocess_prediction(
            pred[pred_field_iou], max_clusters_for_dilation=1e20)  # highest precision
        preds.append(pred_l)

        if 1:
            fig, ax = plt.subplots(1, len(pred)+1, figsize=(50, 50))
            plt.tight_layout()
            ax[0].title.set_text('img')
            ax[0].title.set_fontsize(100)
            ax[0].imshow(img)
            #pred_l, _ = postprocess_prediction(pred['seg'], max_clusters_for_dilation=50)
            #ax[1].title.set_text('postproc')
            #ax[1].title.set_fontsize(100)
            #ax[1].imshow(pred_l)
            for j,(name,pred_part) in enumerate(pred.items()):
                mi = pred_part.data.min()
                ma = pred_part.data.max()
                me = pred_part.data.mean()

                pred_part = torch_img_to_numpy(pred_part)
                ax[j+1].title.set_text('%s [%.3g %.3g %.3g]' % (name, mi, me, ma))
                ax[j+1].title.set_fontsize(100)
                ax[j+1].imshow(pred_part)
                fig.savefig(
                    os.path.join(args.out_dir, 'img_%s.%d.png' % (args.experiment, i)))
                #fig.savefig(
                #    os.path.join(args.out_dir, 'img_%s.%s.png' % (args.experiment, str(dset.data_df['size'].iloc[i]))))
            plt.close()

    dset.data_df['pred'] = preds

    dset.data_df['rles'] = dset.data_df['pred'].map(
        lambda x: list(labels_to_rles(x)))

    out_pred_list = []
    for _, c_row in tqdm(dset.data_df.iterrows()):
        for c_rle in c_row['rles']:
            out_pred_list.append({'ImageId': c_row['id'],
                                  'EncodedPixels': ' '.join(np.array(c_rle).astype(str))})

    out_pred_df = pd.DataFrame(out_pred_list)
    msg = '%d regions found for %d images; writing predictions to %s' % (
        out_pred_df.shape[0], dset.data_df.shape[0], args.predictions_file)
    logging.info(msg)
    print msg
    out_pred_df[['ImageId', 'EncodedPixels']].to_csv(
        args.predictions_file, index=False)


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
    for global_state['it'], (img, (labels, labels_bin)) in tqdm(
            enumerate(train_loader, global_state['it'] + 1)):
        model.prepare_fit(img, labels, labels_bin)

        cnt = cnt + 1

        if cnt > 0 and global_state['it'] % eval_every == 0:
            model.fit()
            l = validate_knn(model, valid_loader, criterion)
            img, mask, boundary, blend = model.predict(img)
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


def train_cnn(train_loader,
              valid_loader,
              targets,
              model,
              optimizer,
              scheduler,
              eval_every,
              print_every,
              save_every,
              global_state):

    time_start = time.time()
    time_valid = Meter()

    epoch = global_state['epoch']

    is_lbfgs = global_state['args'].optim == 'lbfgs'

    # lbfgs has to be called with a closure, in contrast to other optimizers
    # the closure() helper function also does gradient accumulation, using this buffer.
    acc = []

    # NOTE on logging: the naming scheme is <train/valid>_<avg/std/last>_<what>. The overall loss is 'loss',
    # which is  the sum of all specified targets. The partial targets are also available under their configured
    # name. 'avg' and 'std' refer to an epoch, 'last' is recorded after each minibatch.

    stats_train = NamedMeter()
    stats_valid = NamedMeter()

    # NOTE on python weirdness: using scalar inside closure gives error! Therefore using arrays with one element
    # https://stackoverflow.com/questions/4851463/python-closure-write-to-variable-in-parent-scope

    closure_cnt = [0]  # (only) for lbfgs, closure can be called multiple times

    def closure():
        optimizer.zero_grad()
        logging.debug('start closure %d' % closure_cnt[0])

        for mb_acc in acc:

            transfer_data(mb_acc, targets, global_state['args'].input_field, global_state['args'].instance_weights)

            pred = run_model(model, mb_acc[global_state['args'].input_field], train=True)

            total_loss = apply_criteria(mb_acc,
                                        pred,
                                        targets,
                                        stats_train,
                                        instance_weight_field=global_state['args'].instance_weights,
                                        use_class_weights=True,
                                        calc_iou=True,
                                        pred_field_iou='seg',
                                        target_field_iou='masks_prep',
                                        max_clusters_for_dilation=50)

            logging.debug('loss: %.3g', as_py_scalar(total_loss))

            total_loss.backward()

        if global_state['args'].clip_gradient > 0:
            gradi = torch.nn.utils.clip_grad_norm(
                model.parameters(), global_state['args'].clip_gradient)
            stats_train.update('grad', gradi)

        closure_cnt[0] += 1

        return total_loss


    acc = []

    total_batches = len(train_loader)
    it_start = global_state['it']
    it_last = it_start + total_batches
    for global_state['it'], mb in tqdm(enumerate(
            train_loader, global_state['it'] + 1), desc='train', total=total_batches):

        it = global_state['it']

        acc.append(mb)
        # last grad accum can be incomplete
        if len(acc) < global_state['args'].grad_accum and it < it_last:
            continue

        closure_cnt = [0]

        # learn!
        model.train()
        if not is_lbfgs:
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)

        num_acc = len(acc)
        acc = []
        train_loss = stats_train['loss'].last

        if math.isnan(train_loss):
            msg = 'iteration %d - training blew up ...' % it
            logging.error(msg)
            raise TrainingBlowupError(msg)

        validated = False  # when doing grad accum, don't print validation results twice
        for i in range(it - num_acc + 1, it + 1):
            if i % eval_every == 0 and not validated:
                if  global_state['args'].cuda > 0:
                    stats_train.update('gpu_mem', get_gpu_used_memory())
                stats_valid.reset()
                # don't apply instance and class weights during evaluation
                validate(stats_valid, valid_loader, targets, model, global_state['args'].input_field,
                         instance_weight_field=None, use_class_weights=False, calc_iou=True)
                time_valid.update(stats_valid['time'])
                for k,v in stats_valid.items():
                    insert_log(i, 'valid_avg_%s' % k, v.avg)
                    insert_log(i, 'valid_std_%s' % k, v.std)
                validated = True

            iou = get_latest_log('valid_avg_iou', float('nan'))[0]
            l = get_latest_log('valid_avg_loss', float('nan'))[0]

            if i % print_every == 0:

                logging.info('[%d, %d]\ttrain loss: %.3f\tvalid loss: %.3f\tvalid iou: %.3f\tlr: %g' %
                    (epoch, i, train_loss, l, iou, global_state['lr']))
                save_plot(os.path.join(global_state['args'].out_dir, 'progress.png'), global_state['args'].experiment)

            if i % save_every == 0:
                is_best = False
                # smooth values over iterations
                if iou > 0.0:
                    h = get_log('valid_avg_iou')
                    n = min(5, len(h[0]))
                    m = moving_average(h[0], n)
                    cur = m[-1]
                    if global_state['best_iou'] < cur:
                        global_state['best_iou'] = cur
                        global_state['best_iou_it'] = global_state['it']
                        is_best = True
                        logging.info(
                            '[%d, %d]\t new best: it = %d, loss = %.5f, iou = %.5f' %
                            (epoch, i, global_state['it'], l, cur))

                save_checkpoint(
                    get_checkpoint_file(global_state['args']), model, optimizer, global_state, is_best)

        for k,v in stats_train.items():
            insert_log(it, 'train_last_%s' % k, v.last)

    time_end = time.time()
    time_total = time_end - time_start
    stats_train.update('time', time_total)

    for k,v in stats_train.items():
        insert_log(it, 'train_avg_%s' % k, v.avg)
        insert_log(it, 'train_std_%s' % k, v.std)

    insert_log(it, 'lr', global_state['lr'])

    return stats_train, stats_valid


def make_criterion(args):
    """create a training criterion"""
    if args.instance_weights is not None and args.criterion != 'bce':
            raise ValueError(
                'instance weights currently only supported for bce criterion')
    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'bce':
        if args.instance_weights is not None:
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
    return criterion


def epoch_logging_message(global_state, targets, stats_train, stats_valid, len_train=None, len_valid=None):

    msg = '[%d, %d]' % (global_state['epoch'], global_state['it'])
    msg += '\tTRAIN loss = %.3g +- %.3g\tiou = %.3g += %.3g' % (stats_train['loss'].avg, stats_train['loss'].std, stats_train['iou'].avg, stats_train['iou'].std)

    for k in [target['name'] for target in targets]:
        msg += '\t%s = %.3g +- %.3g' % (k, stats_train[k].avg, stats_train[k].std)

    msg += '\tVAL loss = %.3g +- %.3g\tiou = %.3g +- %.3g' % (stats_valid['loss'].avg, stats_valid['loss'].std, stats_valid['iou'].avg, stats_valid['iou'].std)

    for target in targets:
        name = target['name']
        msg += '\t%s = %.3g +- %.3g' % (name, stats_valid[name].avg, stats_valid[name].std)

    msg += '\twt = %.3g +- %.3g' % (stats_train['inst_wt'].avg, stats_train['inst_wt'].std)

    msg += '\tlr = %.3g' % (global_state['lr'])

    if len_train is not None and len_valid is not None:
        time_total = stats_train['time'].sum
        time_val = stats_valid['time'].sum
        n_val = stats_valid['time'].count

        msg += '\tepoch time=%d\tval time=%d\t sec/ex=%.2f\t train sec/ex=%.2f\tvalid sec/ex=%.2f' % (time_total, time_val, 1.0 * time_total / len_train, 1.0 * (time_total - time_val) / len_train, 1.0 * time_val / (n_val * len_valid) if n_val * len_valid > 0 else 0.0)

    if  global_state['args'].cuda > 0 and stats_train['gpu_mem'].count > 0:
        msg += '\tgpu_mem=%d' % stats_train['gpu_mem'].avg

    return msg


def init_cuda(benchmark):
    if not torch.cuda.is_available():
        raise ValueError('cuda requested, but not available')

    # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    # note: actually makes it a lot slower on this problem!
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.enabled = True

    global dev
    def dev(x): return x.cuda()

    print '\tset cuda environment'
    print '\t\ttorch.__version__              =', torch.__version__
    print '\t\ttorch.version.cuda             =', torch.version.cuda
    print '\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version()
    try:
        NUM_CUDA_DEVICES = len(
            os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        print '\t\tos[\'CUDA_VISIBLE_DEVICES\']  =', os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        print '\t\tos[\'CUDA_VISIBLE_DEVICES\']  =', 'None'
        NUM_CUDA_DEVICES = 1

    print '\t\ttorch.cuda.device_count()   =', torch.cuda.device_count()
    print '\t\ttorch.cuda.current_device() =', torch.cuda.current_device()


# https://stackoverflow.com/questions/49595663/find-a-gpu-with-enough-memory
def get_gpu_used_memory():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map[torch.cuda.current_device()]


def parse_targets(args):
    targets = []

    for spec in args.targets:
        # <name>:<dataset column>:<criterion_weight>:<class_weight>
        # criterion and class weights are optional
        try:
            parts = spec.split(':')
            if len(parts) == 2:
                w_crit = 1.0
                w_class = 1.0
            elif len(parts) == 3:
                w_crit = float(parts[2])
                w_class = 1.0
            elif len(parts) == 4:
                if len(parts[2]) == 0:
                    w_crit = 1.0
                else:
                    w_crit = float(parts[2])
                w_class = float(parts[3])
            else:
                raise ValueError('invalid target specification: %s' % spec)
        except:
            raise ValueError('invalid target specification: %s' % spec)

        targets.append(
            {'name' : parts[0],
             'col' : parts[1],
             'crit' : make_criterion(args),
             'w_crit' : w_crit,
             'w_class': w_class})

    # normalize weights
    s = float(sum([target['w_crit'] for target in targets]))
    for target in targets:
        target['w_crit'] /= s

    return targets

def main():
    parser = configargparse.ArgumentParser( description='training and testing of NN model.')
    parser.add( '--config', '-c', default='default.cfg', is_config_file=True, help='config file path [default: %(default)s])')
    parser.add( '--model', help='cnn/knn', choices=[ 'knn', 'cnn'], required=True, default="")

        # parser.add('--arch', '-a', metavar='ARCH', default='resnet18',
        #                                        choices=model_names,
        #                                        help='model architecture: ' +
        #                                            ' | '.join(model_names) +
        #                                            ' (default: resnet18)')
    parser.add('--experiment', '-e', required=True, help='experiment name')
    parser.add('--out-dir', '-o', help='output directory')
    parser.add('--resume', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add( '--override-model-opts', type=csv_list, default='override-model-opts,resume,experiment,out-dir,save-every,print-every,eval-every,scheduler,log-file,do,stop-instance-after', help='when resuming, change these options [default: %(default)s]')
    parser.add('--force-overwrite', type=int, default=0, help='overwrite existing checkpoint, if it exists [default: %(default)s]')
    parser.add( '--do', choices=('train','score','submit','baseline'), default='train', help='mode of operation. score: compute losses and iou over training and validation sets. submit: write output files with run-length encoded predictions. baseline: compute losses with global average as prediction [default: %(default)s]')
    parser.add( '--predictions-file', type=str, help='file name for predictions output')
    parser.add('--data', '-d', metavar='DIR', required=True, help='path to dataset')
    parser.add('--stage', '-s', default='stage1', help='stage [default: %(default)s]')
    parser.add('--group', '-g', default='train', help='group name [default: %(default)s]')
    parser.add('--crop-size', type=int_list, default='192,192', help='crop images to this size during training [default: %(default)s]')
    parser.add('--valid-fraction', '-v', default=0.25, type=float, help='validation set fraction [default: %(default)s]')
    parser.add( '--stratify', type=int, default=1, help='stratify train/test split according to image size [default: %(default)s]')
    parser.add('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run [default: %(default)s]')
    parser.add('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size [default: %(default)s]')
    parser.add( '--grad-accum', default=1, type=int, metavar='N', help='number of batches between gradient descent [default: %(default)s]')
    parser.add('--init-weight', default='kaiming', choices=['kaiming', 'xavier', 'default'], help='weight initialization method default: %(default)s]')
    parser.add('--init-output-bias', default='', type=float_dict, help='initialize biases of output layers to match average value, in the form <target name1>:<value1>,<target name2>:<value2>,... ')
    parser.add( '--input-field', type=str, default='images_prep', help='dataset field to pass to model as input [default: %(default)s]')
    parser.add( '--targets', type=csv_list, default='seg:masks_prep_bin:1.0:1.0', help='one or multiple targets, as comma-delimited list of: <name>:<dataset field>:<target_weight>:<class_weight>. class_weight is a weight multiplier for non-zero mask pixels. <target_weight> and <class_weight> are optional. For multiple targets, model is expected to return dictionary of target names [default: %(default)s]')
    parser.add( '--criterion', '-C', default='bce', choices=[ 'mse', 'bce', 'jaccard', 'dice'], help='type of loss function [default: %(default)s]')
    parser.add( '--instance-weights', type=str,  metavar='W', help='use this dataset column as instance weights during training [default: %(default)s]')
    parser.add('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay [default: %(default)s]')
    parser.add( '--optim', '-O', default='adam', choices=[ 'sgd', 'adam', 'lbfgs'], help='optimization algorithm [default: %(default)s]')
    parser.add( '--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate [default: %(default)s]')
    parser.add('--momentum', '-m', default=0.9, type=float, metavar='M', help='momentum [default: %(default)s]')
    parser.add('--history-size', type=int, default=100, help='history size for lbfgs [default: %(default)s]')
    parser.add('--max-iter-lbfgs', type=int, default=20, help='maximum iterations for lbfgs [default: %(default)s]')
    parser.add( '--tolerance-change', type=float, default=0.01, help='tolerance for termination for lbfgs [default: %(default)s]')
    parser.add( '--scheduler', default='none', choices=[ 'none', 'plateau', 'exp', 'multistep'], help='learn rate scheduler [default: %(default)s]')
    parser.add('--lr-decay', default=.1, type=float, metavar='N', help='decay factor for lr scheduler [default: %(default)s]')
    parser.add('--min-lr', default=0.0001, type=float, metavar='N', help='minimum learn rate for scheduler [default: %(default)s]')
    parser.add( '--patience', default=3, type=int, metavar='N', help='patience for lr scheduler, in epochs [default: %(default)s]')
    parser.add('--cooldown', default=5, type=int, metavar='N', help='cooldown for lr scheduler [default: %(default)s]')
    parser.add( '--patience-threshold', default=.1, type=float, metavar='N', help='patience threshold for lr scheduler [default: %(default)s]')
    parser.add( '--scheduler_milestones', type=int_list, default='200', help='list of epoch milestones for multistep scheduler')
    parser.add( '--switch-to-lbfgs', default=0, type=int, metavar='N', help='if lr scheduler reduces rate, switch to lbfgs [default: %(default)s]')
    parser.add( '--clip-gradient', default=0.25, type=float, metavar='C', help='clip excessive gradients during training [default: %(default)s]')
    parser.add('--print-every', '-p', default=20, type=int, metavar='N', help='print frequency [default: %(default)s]')
    parser.add('--save-every', '-S', default=50, type=int, metavar='N', help='save frequency [default: %(default)s]')
    parser.add('--eval-every', default=100, type=int, metavar='N', help='eval frequency [default: %(default)s]')
    parser.add('--random-seed', type=int, default=2018, help='set random number generator seed [default: %(default)s]')
    parser.add('--verbose', '-V', type=int, default=0, help='verbose logging')
    parser.add('--log-file', help='write logging output to file')
    parser.add('-j', '--workers', default=1, type=int, metavar='N', help='number of data loader workers [default: %(default)s]')
    parser.add('--cuda', type=int, default=0, help='use cuda [default: %(default)s]')
    parser.add( '--cuda-benchmark', type=int, default=0, help='use cuda benchmark mode [default: %(default)s]')
    parser.add('--stop-instance-after',metavar='SEC', type=int, default=2592000, help='wen running on AWS, stop the instance after that many seconds')

    args = parser.parse_args()

    # in overrides, replace '-' by '_', and check that it is indeed an option
    new_overrides = []
    for opt in args.override_model_opts:
        opt_new = opt.replace('-', '_')
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

    time_start = time.time()

    if args.predictions_file is None:
        args.predictions_file = os.path.join(
            args.out_dir, 'predictions_%s.csv' %
            args.experiment)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    global_state = {'epoch': -1,
                    'it': -1,
                    'best_loss': 1e20,
                    'best_it': 0,
                    'best_iou': 0.0,
                    'best_iou_it': 0,
                    'lr': args.lr,
                    'args': args}

    # parse target(s)

    targets = parse_targets(args)

    # init cuda

    if args.cuda > 0:
        init_cuda(args.cuda_benchmark>0)

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
        #model = UNetClassify(layers=4, init_filters=16)
        model = UNetClassifyMulti(targets, layers=4, init_filters=16)
        if args.init_weight != 'default' or len(args.init_output_bias) > 0:
            init_weights(model, args.init_weight, args.init_output_bias)
        model = dev(model)
    else:
        raise ValueError("Only supported models are cnn or knn")

    # optionally resume from a checkpoint
    if args.do != 'train' and args.resume is None:
        raise ValueError('--resume must be specified')

    if args.resume is not None:
        model = load_checkpoint(
            checkpoint_file_from_dir(
                args.resume), model, None, global_state)
        # make sure args here is consistent with possibly updated
        # global_state['args']!
        args = global_state['args']

        # if targets from model are not overriden, reparsing is necessary
        if 'targets' not in args.override_model_opts:
            targets = parse_targets(args)

        args.force_overwrite = 1
        model = dev(model)
    else:
        # prevent accidental overwriting
        ckpt_file = get_checkpoint_file(args)
        if os.path.isfile(ckpt_file) and args.force_overwrite == 0:
            raise ValueError(
                'checkpoint already exists, exiting: %s' %
                ckpt_file)
        clear_log()

    logging.info('model:\n')
    logging.info(model)
    logging.info('number of parameters: %d\n' %
                 sum([param.nelement() for param in model.parameters()]))

    # set up optimizer

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(),
                                lr=args.lr,
                                max_iter=args.max_iter_lbfgs,
                                history_size=args.history_size,
                                tolerance_change=args.tolerance_change)
    else:
        raise ValueError('unknown optimization: %s' % args.optim)


    # set up learn rate scheduler

    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau2(
            optimizer,
            factor=args.lr_decay,
            patience=args.patience,
            patience_threshold=args.patience_threshold,
            cooldown=args.cooldown,
            min_lr=args.min_lr,
            verbose=1)
    elif args.scheduler == 'multistep':
        if args.scheduler_milestones is None or len(
                args.scheduler_milestones) == 0:
            raise ValueError(
                'scheduler-milestones cannot be empty for multi-step')
        scheduler = MultiStepLR(optimizer, args.scheduler_milestones)
    elif args.scheduler == 'exp':
        # dummy for now
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_decay)


    # load data

    if args.do == 'train':
        dset_type = 'train'
    elif args.do == 'submit':
        dset_type = 'test'
    else:
        dset_type = 'valid'

    def load_data():
        return NucleusDataset(
            args.data,
            stage_name=args.stage,
            group_name=args.group,
            dset_type=dset_type,
            crop_size=args.crop_size)
    timer = timeit.Timer(load_data)
    t, dset = timer.timeit(number=1)
    logging.info('load time: %.1f\n' % t)


    # which fields should the data loader return?

    fields_test = [args.input_field]
    fields_valid = [args.input_field]
    for target in targets:
        fields_valid.append(target['col'])
    if not 'masks_prep' in fields_valid:
        fields_valid.append('masks_prep') # for iou
    fields_train = [x for x in fields_valid]
    if args.instance_weights is not None:
        fields_train.append(args.instance_weights)

    # make submission file

    if args.do == 'submit':
        make_submission(dset, model, args)
        return


    # split data

    batch_size_train = args.batch_size
    if args.do != 'train':
        # originals have varying dimensions, but minibatching requires identical sizes
        batch_size_train = 1
    batch_size_valid = 1
    train_dset, valid_dset = dset.train_test_split(
        test_size=args.valid_fraction, random_state=args.random_seed, shuffle=True, stratify=(args.stratify>0))

    if args.do in ('score', 'baseline'):
        train_dset.dset_type = 'valid'

    train_dset.preprocess()
    valid_dset.preprocess()

    train_loader = DataLoader(train_dset, batch_size=batch_size_train, shuffle=True,
                              pin_memory=(args.cuda > 0), num_workers=args.workers)
    valid_loader = DataLoader(valid_dset, batch_size=batch_size_valid, shuffle=True,
                               pin_memory=(args.cuda > 0), num_workers=args.workers)

    logging.info('train set size: %d; test set size: %d' % (len(train_dset), len(valid_dset)))


    # score data

    if args.do in ('score', 'baseline'):

        train_dset.return_fields = fields_valid
        valid_dset.return_fields = fields_valid

        max_clusters =1e20 if args.do == 'score' else 100

        stats_train = NamedMeter()
        validate(stats_train, train_loader, targets, model, global_state['args'].input_field, instance_weight_field=None,
                 use_class_weights=False, calc_iou = (args.do == 'score'), max_clusters_for_dilation=max_clusters, calc_baseline = (args.do == 'baseline'), desc='train')

        stats_valid = NamedMeter()
        validate(stats_valid, valid_loader, targets, model, global_state['args'].input_field, instance_weight_field=None,
                 use_class_weights=False, calc_iou = (args.do == 'score'), max_clusters_for_dilation=max_clusters, desc='valid')
        msg = epoch_logging_message(global_state, targets, stats_train, stats_valid)
        logging.info(msg)
        print msg
        return


    # run training

    train_dset.return_fields = fields_train
    valid_dset.return_fields = fields_valid

    logging.info('command line options:\n')
    for k in global_state['args'].__dict__:
        logging.info('> %s = %s' % (k, str(global_state['args'].__dict__[k])))
    logging.info('')

    recovered_ckpt = None
    recovery_attempts = 0
    last_epoch_loss = get_latest_log('train_avg_loss', 1e20)[0]

    for global_state['epoch'] in range(global_state['epoch'] + 1, args.epochs):
        epoch = global_state['epoch']
        try:
            stats_train, stats_valid = trainer(train_loader, valid_loader, targets, model, optimizer, scheduler, args.eval_every, args.print_every, args.save_every, global_state)

            msg = epoch_logging_message(global_state, targets, stats_train, stats_valid, len(train_dset), len(valid_dset))

            logging.info(msg)

            # check for blowup
            epoch_loss = get_latest_log('train_avg_loss', 1e20)[0]
            if not math.isnan(last_epoch_loss) and epoch_loss > 100.0 * last_epoch_loss:
                msg = 'iteration %d - training blew up ...' % it
                logging.error(msg)
                raise TrainingBlowupError(msg)
            last_epoch_loss = epoch_loss

            save_checkpoint(
                get_checkpoint_file(global_state['args'], global_state['it']),
                model,
                optimizer,
                global_state)


            # check elapsed time
            time_end = time.time()
            time_total = time_end - time_start
            logging.info('[%d, %d] total running time: %d seconds' % (global_state['epoch'], global_state['it'], time_total))
            if time_total > args.stop_instance_after:
                ret = stop_current_instance(False)
                logging.info(ret)
                return 0

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
                    if not args.switch_to_lbfgs or isinstance(
                            optimizer, optim.LBFGS):
                        logging.info(
                            '[%d, %d]\tLR changed from %f to %f.' %
                            (epoch, global_state['it'], lr_old, lr_new))
                        global_state['lr'] = lr_new
                    else:
                        logging.info('[%d, %d]\tswitching to lbfgs' %
                                     (epoch, global_state['it']))
                        lr = 0.8
                        optimizer = optim.LBFGS(
                            model.parameters(),
                            lr=lr,
                            max_iter=args.max_iter_lbfgs,
                            history_size=args.history_size,
                            tolerance_change=args.tolerance_change)
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
                    msg = 'attempt %d: using lbfgs and lr (%f) already at min_lr (%f), giving up' % (
                        recovery_attempts, global_state['lr'], min_lr)
                    logging.error(msg)
                    raise
                model = load_checkpoint(ckpt,
                                        model,
                                        optimizer,
                                        global_state)
                global_state['lr'] = lr
                optimizer = optim.LBFGS(model.parameters(),
                                        lr=lr,
                                        max_iter=args.max_iter_lbfgs,
                                        history_size=args.history_size,
                                        tolerance_change=args.tolerance_change)
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

                logging.error(
                    'recovered from checkpoint %s (attempt %d), lr = %f. keeping fingers crossed ...' %
                    (ckpt, recovery_attempts, lr))
            else:
                logging.error('cannot recover ... terminating.')
                raise

    msg = 'done with epoch %d' % global_state['epoch']
    print msg
    logging.info(msg)
    logging.info(msg)
    if args.stop_instance_after > 0:
        ret = stop_current_instance(False)
        logging.info(ret)
        return 0


if __name__ == '__main__':
    main()
