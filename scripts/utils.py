#!/usr/bin/env python

import sys
import traceback
import math
import numpy as np
import os
import logging
import socket
import errno
import pickle

import skimage
from skimage import img_as_float, exposure
from skimage.morphology import label
from skimage.io import imread
#from matplotlib import _cntr as cntr
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import ToTensor, ToPILImage


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
        return traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1])[0].strip()

def clear_log():
    global LOG
    LOG = []

def get_log():
    global LOG
    return LOG

def set_log(l):
    global LOG
    LOG = l

def insert_log(it, k, v):
    global LOG
    #logging.info('INSERT %s %s %s %s' % (it, k, v, type(v)))
    if len(LOG) > 0:
        last = LOG[-1]
        if last['it'] > it:
            msg = 'trying to change history at %d, current is %d' % (it, last['it'])
            logging.error(msg)
            # raise ValueError('trying to change history at %d, current is %d' % (it, last['it']))
            return
        if last['it'] != it:
            last = {'it':it}
            LOG.append(last)
    else:
        last = {'it':it}
        LOG = [last]
    last[k] = v

def get_latest_log(what, default=None):
    global LOG
    latest_row = []
    latest_it = -1
    for row in LOG[::-1]:
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


def get_history_log(what, log=None):
    global LOG
    if not log:
        log=LOG
    vals = [x[what] for x in log if what in x]
    its = [x['it'] for x in log if what in x]
    return vals, its


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
        filename = opts.log_file.replace('$','').format(**os.environ)
    verbose = False
    if hasattr(opts, 'verbose'):
        verbose = opts.verbose>0
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(asctime)s\t%(process)d\t%(levelname)s]\t%(message\
)s", datefmt="%Y%m%d %H:%M:%S", filename=filename)



def strip_end(text, suffix):
    if not text.endswith(suffix):
        return text
    return text[:len(text)-len(suffix)]


# auxiliary class for comma-delimited command line arguments
def csv_list(init):
    l=init.split(',')
    l = [x.strip() for x in l if len(x)]
    return l

def int_list(init):
    l=init.split(',')
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

#https://stackoverflow.com/questions/394770/override-a-method-at-instance-level
def monkeypatch(obj, fn_name, new_fn):
    fn = getattr(obj, fn_name)
    funcType = type(fn)
    setattr(obj, fn_name, funcType(new_fn, fn_name, obj.__class__))

# Example:
#class Dog():
#    def bark(self):
#       print "Woof"

#def new_bark(self):
#    print "Woof Woof"

#foo = Dog()
#foo.bark()
#
#
#monkeypatch(foo, 'bark', new_bark)
#
#foo.bark()

def is_inverted(img,invert_thresh_pd=10.0):
    img_grey = img_as_ubyte(rgb2grey(img))
    img_th = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[1]

    return np.sum(img_th==255)>((invert_thresh_pd/10.0)*np.sum(img_th==0))


def torch_to_numpy(t):
    t = t.numpy().squeeze()
    if t.ndim > 2:
        t = t.transpose(1,2,0)
    t = (t/t.max()*255).astype(np.uint8)
    return t


def numpy_to_torch(n, unsqueeze=False):
    n = np.ascontiguousarray(n)
           # to avoid 'negative stride' error -
           # see https://discuss.pytorch.org/t/problem-with-reading-pfm-image/2924
    if n.ndim == 2:
        n = np.expand_dims(n, 2)
    if n.ndim == 3 and n.shape[2] == 1:
        # single channel or mask
        n_conv = torch.from_numpy(np.transpose(n, (2, 0, 1))).float()
    else:
        n_conv = ToTensor()(n)
    if unsqueeze:
        n_conv = n_conv.unsqueeze(0)
    return n_conv


# RLE encoding

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def check_encoding():
    input_path = '../input/train'
    masks = [f for f in os.listdir(input_path) if f.endswith('_mask.tif')]
    masks = sorted(masks, key=lambda s: int(s.split('_')[0])*1000 + int(s.split('_')[1]))

    encodings = []
    N = 100     # process first N masks
    for i,m in enumerate(masks[:N]):
        if i % 10 == 0: print('{}/{}'.format(i, len(masks)))
        img = Image.open(os.path.join(input_path, m))
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        encodings.append(rle_encoding(x))

    #check output
    conv = lambda l: ' '.join(map(str, l)) # list -> string
    subject, img = 1, 1
    print('\n{},{},{}'.format(subject, img, conv(encodings[0])))

    # train_masks.csv:
    print('1,1,168153 9 168570 15 168984 22 169401 26 169818 30 170236 34 170654 36 171072 39 171489 42 171907 44 172325 46 172742 50 173159 53 173578 54 173997 55 174416 56 174834 58 175252 60 175670 62 176088 64 176507 65 176926 66 177345 66 177764 67 178183 67 178601 69 179020 70 179438 71 179857 71 180276 71 180694 73 181113 73 181532 73 181945 2 181950 75 182365 79 182785 79 183205 78 183625 78 184045 77 184465 76 184885 75 185305 75 185725 74 186145 73 186565 72 186985 71 187405 71 187825 70 188245 69 188665 68 189085 68 189506 66 189926 65 190346 63 190766 63 191186 62 191606 62 192026 61 192446 60 192866 59 193286 59 193706 58 194126 57 194546 56 194966 55 195387 53 195807 53 196227 51 196647 50 197067 50 197487 48 197907 47 198328 45 198749 42 199169 40 199589 39 200010 35 200431 33 200853 29 201274 27 201697 20 202120 15 202544 6')


def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


def labels_to_rles(lab_img):
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


def read_img_join_masks(img_id, root='../../input/stage1_train/'):
    img = imread(os.path.join(root, img_id, 'images', img_id + '.png'))
    path = os.path.join(root, img_id, 'masks')
    mask = None
    i = 0
    for mask_file in next(os.walk(path))[2]:
        if mask_file.endswith('png'):
            i = i + 1
            mask_ = imread(os.path.join(path, mask_file)).astype(int)
            mask_[mask_>0] = i
            if mask is None:
                mask = mask_
            else:
                mask = np.maximum(mask, mask_)
    return img, mask

# https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
# add contour to plot, without directly plotting!
# assuming input is (integer) label mask
def add_contour(z, ax, color='black'):

    z = z.astype(int)
    x, y = np.mgrid[:z.shape[0], :z.shape[1]]
    c = cntr.Cntr(x, y, z)

    # trace a contour at z == 0.5
    for at in range(np.amin(z), np.amax(z)):
        res = c.trace(at + 0.5)

        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(res) // 2
        segments, codes = res[:nseg], res[nseg:]

        for seg in segments:
            # for some reason, the coordinates are flipped???
            p = plt.Polygon([[x[1],x[0]] for x in seg], fill=False, color=color, linewidth=2.0)
            ax.add_artist(p)

# display one or several images
def show_images(img, max_col=3):
    if not isinstance(img, (tuple, list)):
        img = [img]

    l = len(img)
    if l <= max_col:
        ncol = l
        nrow = 1
    else:
        ncol = max_col
        nrow = int(math.ceil(1.0*l/ncol))

    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 16))
    plt.tight_layout()
    for i in range(l):
        if l == 1:
            axi = ax
        elif l <= max_col:
            axi = ax[i]
        else:
            c = int(math.floor(1.0*i/ncol))
            r = i - c * max_col
            axi = ax[c,r]

        axi.grid(None)
        if isinstance(img[i], np.ndarray):
            img_conv = img[i]
        else:
            img_conv =  ToPILImage()(img[i])
        #if isinstance(img_conv, torch.Tensor):
        #    sz = img_conv.size()
        #    if len(sz) == 3 and sz[0] == 1:
        #        img_conv = img_conv.squeeze().numpy()
        #    else:
        #        img_conv = img_conv.permute(2,3,1,0).squeeze().numpy()
        axi.imshow(img_conv,cmap='gray')
    if l > max_col:
        for i in range(l,max_col*int(math.ceil(1.0*l/max_col))):
            c = int(math.floor(1.0*i/ncol))
            r = i - c * max_col
            ax[c,r].axis('off')
    plt.show()

def show_with_contour(img, mask, color='black'):
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(img)
    add_contour(mask, ax, color)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()

# (copied from tutorial)
def plot_img_and_hist(image, axes=None, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """

    if isinstance(image, torch.Tensor):
        image = image.numpy()
    image = image.squeeze()

    if axes is None:
        fig,axes = plt.subplots(1,2,figsize=(16, 16))
        plt.tight_layout()

    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap='gray')
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])
    ax_hist.grid(None)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

if __name__ == '__main__':
    pass
