#!/usr/bin/env python

import sys
import traceback
import numpy as np
import os
import logging
import socket
import errno
import pickle
import re
import urllib2
import boto3

from PIL import Image

from skimage.morphology import label

import torch


def as_py_scalar(x):
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                      torch.cuda.ByteTensor, torch.cuda.IntTensor, torch.cuda.LongTensor)):
        x = x.cpu()
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.item()
    return x


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    assert(len(lr) == 1)  # we support only one param_group
    lr = lr[0]

    return lr


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def exceptions_str():
    return traceback.format_exception_only(
        sys.exc_info()[0], sys.exc_info()[1])[0].strip()


LOG = []

def clear_log():
    global LOG
    LOG = []


def get_the_log():
    global LOG
    return LOG


def set_the_log(l):
    global LOG
    LOG = l


def insert_log(it, k, v, log=None):
    #logging.info('INSERT %s %s %s %s' % (it, k, v, type(v)))

    global LOG
    if log is None:
        if LOG is None:
            LOG = []
        log = LOG

    v = as_py_scalar(v)

    if len(log) > 0:
        last = log[-1]
        if last['it'] > it:
            msg = 'trying to change history at %d, current is %d' % (it, last['it'])
            logging.error(msg)
            # raise ValueError('trying to change history at %d, current is %d' % (it, last['it']))
            return
        if last['it'] != it:
            last = {'it': it}
            log.append(last)
    else:
        last = {'it': it}
        log.append(last)
    last[k] = v


def get_latest_log(what, default=None, log=None):
    global LOG
    if log is None:
        log = LOG

    latest_row = []
    latest_it = -1
    for row in log[::-1]:
        if row['it'] > latest_it and what in row:
            latest_it = row['it']
            latest_row = row

    if latest_it == -1 or what not in latest_row:
        if default is not None:
            return default, 0
        msg = 'no such key in log: %s' % what
        logging.error(msg)
        raise ValueError('no such key in log: %s' % what)

    return latest_row[what], latest_it


def get_log(what, log=None):
    global LOG
    if log is None:
        log = LOG
    vals = [x[what] for x in log if what in x]
    its = [x['it'] for x in log if what in x]
    return vals, its


def list_log_keys(log=None):
    global LOG
    if log is None:
        log = LOG
    keys = set()
    for entry in log:
        keys |= set(entry.keys())
    return sorted(keys)


def get_checkpoint_file(args, it=0):
    if it > 0:
        return os.path.join(
            args.out_dir, 'model_save_%s.%d.pth.tar' %
            (args.experiment, it))
    return os.path.join(
        args.out_dir,
        'model_save_%s.pth.tar' %
        args.experiment)


def get_latest_checkpoint_file(args):
    last_it = -1
    last_ckpt = ''
    pattern = re.compile(
        'model_save_%s.(?P<it>[0-9]+).pth.tar' %
        args.experiment)
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


def checkpoint_file_from_dir(fname):
    if os.path.isfile(fname):
        return fname
    if not os.path.isdir(fname):
        raise ValueError('checkpoint not found: %s', fname)
    exp_name = filter(lambda x: len(x) > 0, fname.split('/'))[-1]
    guess = os.path.join(fname, 'model_save_%s.pth.tar' % exp_name)
    if not os.path.isfile(guess):
        raise ValueError('checkpoint not found: %s', fname)
    return guess


def init_logging(opts={}):
    filename = None
    if hasattr(opts, 'log_file') and opts.log_file:
        # slight hack: dollar signs work if config file is read by shell, here we need to strip it
        # another slight hack: sometimes HOSTNAME is not set
        if 'HOSTNAME' not in os.environ.keys():
            os.environ['HOSTNAME'] = socket.gethostname()
        filename = opts.log_file.replace('$', '').format(**os.environ)
    verbose = False
    if hasattr(opts, 'verbose'):
        verbose = opts.verbose > 0
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(asctime)s\t%(process)d\t%(levelname)s]\t%(message\
)s", datefmt="%Y%m%d %H:%M:%S", filename=filename)


def strip_end(text, suffix):
    if not text.endswith(suffix):
        return text
    return text[:len(text) - len(suffix)]


# auxiliary class for comma-delimited command line arguments
def csv_list(init):
    l = init.split(',')
    l = [x.strip() for x in l if len(x)]
    return l


def int_list(init):
    l = init.split(',')
    l = [int(x.strip()) for x in l if len(x)]
    return l


def mkdir_p(path):
    try:
        os.makedirs(path)
        return 0
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return 1
        else:
            raise

# https://stackoverflow.com/questions/394770/override-a-method-at-instance-level


def monkeypatch(obj, fn_name, new_fn):
    fn = getattr(obj, fn_name)
    funcType = type(fn)
    setattr(obj, fn_name, funcType(new_fn, fn_name, obj.__class__))

# Example:
# class Dog():
#    def bark(self):
#       print "Woof"

# def new_bark(self):
#    print "Woof Woof"

#foo = Dog()
# foo.bark()
#
#
#monkeypatch(foo, 'bark', new_bark)
#
# foo.bark()


# RLE encoding

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[
        0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def check_encoding():
    input_path = '../input/train'
    masks = [f for f in os.listdir(input_path) if f.endswith('_mask.tif')]
    masks = sorted(masks, key=lambda s: int(
        s.split('_')[0]) * 1000 + int(s.split('_')[1]))

    encodings = []
    N = 100     # process first N masks
    for i, m in enumerate(masks[:N]):
        if i % 10 == 0:
            print('{}/{}'.format(i, len(masks)))
        img = Image.open(os.path.join(input_path, m))
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        encodings.append(rle_encoding(x))

    # check output
    def conv(l):
        return ' '.join(map(str, l))  # list -> string
    subject, img = 1, 1
    print('\n{},{},{}'.format(subject, img, conv(encodings[0])))

    # train_masks.csv:
    print('1,1,168153 9 168570 15 168984 22 169401 26 169818 30 170236 34 170654 36 171072 39 171489 42 171907 44 172325 46 172742 50 173159 53 173578 54 173997 55 174416 56 174834 58 175252 60 175670 62 176088 64 176507 65 176926 66 177345 66 177764 67 178183 67 178601 69 179020 70 179438 71 179857 71 180276 71 180694 73 181113 73 181532 73 181945 2 181950 75 182365 79 182785 79 183205 78 183625 78 184045 77 184465 76 184885 75 185305 75 185725 74 186145 73 186565 72 186985 71 187405 71 187825 70 188245 69 188665 68 189085 68 189506 66 189926 65 190346 63 190766 63 191186 62 191606 62 192026 61 192446 60 192866 59 193286 59 193706 58 194126 57 194546 56 194966 55 195387 53 195807 53 196227 51 196647 50 197067 50 197487 48 197907 47 198328 45 198749 42 199169 40 199589 39 200010 35 200431 33 200853 29 201274 27 201697 20 202120 15 202544 6')


def prob_to_rles(x, cut_off=0.5):
    lab_img = label(x > cut_off)
    if lab_img.max() < 1:
        lab_img[0, 0] = 1  # ensure at least one prediction per image
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def labels_to_rles(lab_img):
    if lab_img.max() < 1:
        lab_img[0, 0] = 1  # ensure at least one prediction per image
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# Amazon stuff

def get_current_instance_id():
    return urllib2.urlopen('http://169.254.169.254/latest/meta-data/instance-id').read()


def stop_current_instance(dry_run=True):
    instance_id = get_current_instance_id()
    ec2 = boto3.resource('ec2')
    msg = 'instance shutting down'
    print msg
    logging.warning(msg)
    ret = ec2.instances.filter(InstanceIds=[instance_id]).stop(DryRun=dry_run)
    return ret

if __name__ == '__main__':
    pass
