#!/usr/bin/env python
'''
Fast inplementation of Run-Length Encoding algorithm
Takes only 200 seconds to process 5635 mask files
'''

import math
import numpy as np
import os
import errno
import skimage
from skimage import img_as_float, exposure
from skimage.io import imread
from matplotlib import _cntr as cntr
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import ToPILImage



def strip_end(text, suffix):
    if not text.endswith(suffix):
        return text
    return text[:len(text)-len(suffix)]

                
# auxiliary class for comma-delimited command line arguments                                                                           
def csv_list(init):
    l=init.split(',')
    l = [x.strip() for x in l if len(x)]
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
def add_contour(z, ax, color='black'):

    z = z.astype(int)
    x, y = np.mgrid[:z.shape[0], :z.shape[1]]
    c = cntr.Cntr(x, y, z)

    # trace a contour at z == 0.5
    for at in range(1, np.amax(z)+1):
        res = c.trace(at)

        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(res) // 2
        segments, codes = res[:nseg], res[nseg:]

        for seg in segments:
            # for some reason, the coordinates are flipped???
            p = plt.Polygon([[x[1],x[0]] for x in seg], fill=False, color=color)
            ax.add_artist(p)

# display one or several images
def show_images(img):
    max_col = 3
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

def show_with_contour(img, mask):
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(img)
    add_contour(mask, ax)
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
    # check_encoding()
    print 'hello'

