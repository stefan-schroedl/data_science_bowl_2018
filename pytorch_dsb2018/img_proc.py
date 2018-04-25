import os
import logging

import math
import numpy as np

import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor, ToPILImage

import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from five_crop_aug import FiveCrop

import skimage
from skimage.color import rgb2grey
from skimage import img_as_ubyte, img_as_float, exposure, morphology
from skimage.io import imread
from skimage.feature import peak_local_max
from skimage.util import invert
from skimage.filters import threshold_otsu

from scipy import ndimage as ndi

import torch

from loss import diagnose_errors

from utils import exceptions_str


######## conversion, io

def torch_img_to_numpy(t):
    if isinstance(t, torch.autograd.Variable):
        t = t.data
    t = t.cpu().numpy().squeeze()
    if t.ndim > 2:
        t = t.transpose(1, 2, 0)
    if t.ndim == 3:
        ax = (0,1)
    else:
        ax = None
    mi = t.min(axis=ax)
    ma = t.max(axis=ax)
    eps = 1e-16
    t = ((t - mi) / (ma - mi + eps) * 255).astype(np.uint8)
    return t


def numpy_img_to_torch(n, unsqueeze=False):

    # to avoid 'negative stride' error -
    # see https://discuss.pytorch.org/t/problem-with-reading-pfm-image/2924
    n = np.ascontiguousarray(n)

    mi = np.min(n)
    ma = np.max(n)

    if mi < 0.0:
        logging.warning('image values out of range: [%f %f]' % (mi, ma))
        n -= mi
    if ma > 255.0:
        logging.warning('image values out of range: [%f %f]' % (mi, ma))
        n = (1.0 * n / np.max(n) * 255.0).astype(np.uint8)

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


def read_img_join_masks(img_id, root='../../input/stage1_train/'):
    img = imread(os.path.join(root, img_id, 'images', img_id + '.png'))
    path = os.path.join(root, img_id, 'masks')
    mask = None
    i = 0
    for mask_file in next(os.walk(path))[2]:
        if mask_file.endswith('png'):
            i = i + 1
            mask_ = imread(os.path.join(path, mask_file)).astype(int)
            mask_[mask_ > 0] = i
            if mask is None:
                mask = mask_
            else:
                mask = np.maximum(mask, mask_)
    return img, mask


######## pre/postprocessing


def binarize(img):
    return (img > 0).astype(np.uint8)


def separate_touching_nuclei(labeled_mask, sz=2):
    struc = morphology.disk(sz)

    img_sum = np.zeros(labeled_mask.shape)
    for j in range(1, labeled_mask.max() + 1):
        m = (labeled_mask == j).astype(np.uint8)
        m = morphology.binary_dilation(m, struc)
        img_sum += m
    ov = np.maximum(0, img_sum - 1)

    mask_corrected = np.where(ov == 0, labeled_mask, 0)
    return mask_corrected, ov


# erode masks for training, dilate the prediction back at the end

def erode_mask(mask, sz=2):
    struc = morphology.disk(sz)
    return np.where(
        ndi.binary_erosion(mask, structure=struc, border_value=1),
        mask,
        0)


def redilate_mask(mask_seg, sz=2, skip_clusters=1e20):
    mask_l, n = ndi.label(mask_seg)
    if n >= skip_clusters:
            # too slow, use shortcut
        return mask_l
    struc = morphology.disk(sz)
    mask_dil = np.zeros(mask_l.shape, dtype=np.int32)

    for cluster in range(1, n + 1):
        cur = (mask_l == cluster)
        cur = morphology.binary_dilation(cur, struc)
        mask_dil = np.where(cur, cluster, mask_dil)
    return mask_dil


# training images and masks can be resized, but validation and test images cannot

def is_inverted(img, invert_thresh_pd=.5):
    """more bright pixels (foreground) than dark pixels (background)"""
    thresh = threshold_otsu(img)
    img_th = img > thresh
    return len(np.where(img_th)[0]) < invert_thresh * img.size


def preprocess_img(img, resize):
    if resize is not None:
        img = transform.resize(img, resize)
        # transform.resize() changes type to float!
        img = img_as_ubyte(img)
    return img


def preprocess_img_grey(img, do_inversion=True, invert_thresh=0.5):
    img = img_as_ubyte(rgb2grey(img))
    if do_inversion:
        thresh = threshold_otsu(img)
        img_th = img > thresh
        if len(np.where(img_th)[0]) < invert_thresh * img.size:
            logging.info('invert')
            img = invert(img)
    return img


def preprocess_mask(img, dset_type='train', resize=None):
    if dset_type == 'train':
        if resize is not None:
            img = transform.resize(img, resize)
            #  transform.resize() changes type to float!
            img = img_as_ubyte(img)
        img = erode_mask(img)
    return img


def postprocess_prediction(pred, sz=2, max_clusters_for_dilation=100, thresh=0.0):
    # input is torch tensor of model prediction
    # output is in numpy format
    if not isinstance(pred, np.ndarray):
        pred_np = pred.data.cpu().numpy().squeeze()
    img_th = (pred_np > thresh).astype(int)

    img_th = redilate_mask(img_th, sz=sz, skip_clusters=max_clusters_for_dilation)
    return img_th, pred_np


### augmentation


def torch_flip(t, axis=-1):
    """
    reflect a tensor image along axis.

    use axis=-1 (-2) for horizontal (vertical) flips
    """

    if axis > len(t.size()):
        return t
    inv_idx = torch.arange(t.size(axis)-1, -1, -1).long()
    return t.index_select(axis, inv_idx)


def torch_rot90(t, k=0):
    """rotate a tensor image clockwise k times by 90 degrees"""

    k = k % 4
    if k == 0:
        return t
    if k == 2:
        return torch_flip(torch_flip(t, -1), -2)
    idx = range(len(t.size()))
    idx[-1], idx[-2] = idx[-2], idx[-1]
    t = t.permute(*idx)
    if k == 1:
        return torch_flip(t, -1)
    if k == 3:
        return torch_flip(t, -2)


def noop_augmentation():
    return iaa.Sequential([iaa.Noop()])


# modified from https://github.com/neptune-ml/open-solution-data-science-bowl-2018/wiki

# WARNING: PiecewiseAffine basically erases contour lines!!!
# iaa.PiecewiseAffine(scale=(0.00, 0.06))

def affine_augmentation(crop_size):
    seq = iaa.SomeOf((1, 2),
                     [iaa.Fliplr(0.5),
                      iaa.Flipud(0.5),
                      iaa.Affine(rotate=(-45, 45), mode='symmetric', order=[0, 1]),
                      iaa.Affine(scale=(.6, 1.4), mode='symmetric', order=[0, 1])])

    if crop_size is None:
        return seq
    return iaa.Sequential([seq, FiveCrop(crop_size)])


def color_augmentation():
    return iaa.Sequential([
        # Color
        iaa.OneOf([
            iaa.Sequential([
                iaa.ChangeColorspace(
                    from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((0, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
            iaa.Sequential([
                iaa.ChangeColorspace(
                    from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(1, iaa.Add((0, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
            iaa.Sequential([
                iaa.ChangeColorspace(
                    from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(2, iaa.Add((0, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
            iaa.WithChannels(0, iaa.Add((0, 100))),
            iaa.WithChannels(1, iaa.Add((0, 100))),
            iaa.WithChannels(2, iaa.Add((0, 100)))
        ])
    ], random_order=True)


######## visualization

def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    contours = skimage.measure.find_contours(img, 0) # outside contour
    for contour in contours:
        contour = contour.astype(np.uint8)
        img_contour[contour[:,0], contour[:,1]] = 1
    return img_contour


def add_contour(img, ax, **kwargs):
    contours = skimage.measure.find_contours(img,0)
    if 'linewidth' not in kwargs:
        kwargs['linewidth']=2
    if 'color' not in kwargs:
        kwargs['color']='k'
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], **kwargs)


def show_with_contour(img, mask, color='black'):
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(img)
    add_contour(mask, ax, color)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()


# display one or several images
def show_images(max_col, *img):

    l = len(img)
    if l <= max_col:
        ncol = l
        nrow = 1
    else:
        ncol = max_col
        nrow = int(math.ceil(1.0 * l / ncol))

    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 16))
    plt.tight_layout()
    for i in range(l):
        if l == 1:
            axi = ax
        elif l <= max_col:
            axi = ax[i]
        else:
            c = int(math.floor(1.0 * i / ncol))
            r = i - c * max_col
            axi = ax[c, r]

        axi.grid(None)
        if isinstance(img[i], np.ndarray):
            img_conv = img[i]
        else:
            img_conv = ToPILImage()(img[i])
        axi.imshow(img_conv, cmap='gray')
    if l > max_col:
        for i in range(l, max_col * int(math.ceil(1.0 * l / max_col))):
            c = int(math.floor(1.0 * i / ncol))
            r = i - c * max_col
            ax[c, r].axis('off')
    plt.show()


# (copied from tutorial)
def plot_img_and_hist(image, axes=None, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """

    if isinstance(image, torch.Tensor):
        image = image.numpy()
    image = image.squeeze()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 16))
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


def show_compare_gt(img, pred, mask, thresh=0.5, **opts):
    #pred_masks = parametric_pipeline(img, **opts)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(rgb2grey(img), alpha=0.5),
    add_contour(mask, ax, 'green')
    add_contour(pred, ax, 'red')
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return diagnose_errors(mask, pred, thresh, print_message=True)


######## original classic processing, 'ali's pipeline'

# same as v1, but using skimage instead of cv2
def parametric_pipeline(img,
                        invert_thresh_pd=.5,
                        circle_size=7,
                        disk_size=10,
                        min_distance=9,
                        use_watershed=False
                        ):
    try:
        circle_size = np.clip(int(circle_size), 1, 30)
        if use_watershed:
            disk_size = np.clip(int(disk_size), 0, 50)
            min_distance = np.clip(int(min_distance), 1, 50)

        # Invert the image in case the objects of interest are in the dark side

        thresh = threshold_otsu(img)
        img_th = img > thresh

        if len(np.where(img_th)[0]) > invert_thresh_pd * img.size:
            img = invert(img)

        # morphological opening (size tuned on training data)
        #circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(circle_size, circle_size))
        circle7 = morphology.disk(circle_size / 2.0)
        img_open = morphology.opening(img, circle7)
        #img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, circle7)
        # return img_open

        thresh = threshold_otsu(img_open)
        img_th = (img_open > thresh).astype(int)

        # second morphological opening (on binary image this time)
        bin_open = morphology.binary_opening(img_th, circle7)
        if not use_watershed:
            return ndi.label(bin_open)[0]

        # WATERSHED
        selem = morphology.disk(disk_size)
        dil = morphology.binary_dilation(bin_open, selem)
        img_dist = ndi.distance_transform_edt(dil)
        local_maxi = peak_local_max(img_dist,
                                    min_distance=min_distance,
                                    indices=False,
                                    exclude_border=False)
        markers = ndi.label(local_maxi)[0]
        cc = morphology.watershed(-img_dist,
                                  markers,
                                  mask=bin_open,
                                  compactness=0,
                                  watershed_line=True)

        return cc
    except BaseException:
        logging.error("Error in parametric pipeline:\n%s" % exceptions_str())
        return np.zeros_like(img)


def parametric_pipeline_v1(img,
                           invert_thresh_pd=10,
                           circle_size=7,
                           disk_size=10,
                           min_distance=9,
                           use_watershed=False
                           ):

    circle_size = np.clip(int(circle_size), 1, 30)
    if use_watershed:
        disk_size = np.clip(int(disk_size), 0, 50)
        min_distance = np.clip(int(min_distance), 1, 50)

    # Invert the image in case the objects of interest are in the dark side

    img_grey = img_as_ubyte(img)
    img_th = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)[1]

    if(np.sum(img_th == 255) > ((invert_thresh_pd / 10.0) * np.sum(img_th == 0))):
        print 'invert'
        img = invert(img)

    # reconstruction with dilation
    # best h value = 0.87
    #img_float = img_as_float(img)
    #seed = img_float - h
    #dilated = reconstruction(seed, img_float, method='dilation')
    #hdome = img_float - dilated
    # print 'hdome', hdome.min(), hdome.max()
    # print 'image', img.min(), img.max()
    # print 'dilated', dilated.min(), dilated.max()
    # show_images([img,dilated,hdome])
    hdome = img

    # NOTE: the ali pipeline does the opening before flipping, makes a difference
    # at the edges!

    # morphological opening (size tuned on training data)
    circle7 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (circle_size, circle_size))
    img_open = cv2.morphologyEx(hdome, cv2.MORPH_OPEN, circle7)

    # thresholding
    # isodata thresholding is comparable to otsu ...
    # Otsu thresholding

    img_grey = img_as_ubyte(img_open)
    #th = skimage.filters.threshold_isodata(img_grey)
    #img_th = cv2.threshold(img_grey,th,255,cv2.THRESH_BINARY)[1]
    img_th = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)[1]

    # second morphological opening (on binary image this time)
    bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    if not use_watershed:
        return cv2.connectedComponents(bin_open)[1]

    # WATERSHED
    selem = morphology.disk(disk_size)
    dil = morphology.binary_dilation(bin_open, selem)
    img_dist = ndi.distance_transform_edt(dil)
    local_maxi = peak_local_max(img_dist,
                                min_distance=min_distance,
                                indices=False,
                                exclude_border=False)
    markers = ndi.label(local_maxi)[0]
    cc = morphology.watershed(-img_dist,
                              markers,
                              mask=bin_open,
                              compactness=0,
                              watershed_line=True)

    return cc


# original, verified to be the same as ali
def parametric_pipeline_orig(img_green,
                             invert_thresh_pd=10,
                             circle_size_x=7,
                             circle_size_y=7,
                             ):
    circle_size_x = np.clip(int(circle_size_x), 1, 30)
    circle_size_y = np.clip(int(circle_size_y), 1, 30)

    # green channel happends to produce slightly better results
    # than the grayscale image and other channels
    # morphological opening (size tuned on training data)
    circle7 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (circle_size_x, circle_size_y))
    img_open = cv2.morphologyEx(img_green, cv2.MORPH_OPEN, circle7)
    # Otsu thresholding
    img_th = cv2.threshold(img_open, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th == 255) > ((invert_thresh_pd / 10.0) * np.sum(img_th == 0))):
        img_th = cv2.bitwise_not(img_th)
    # second morphological opening (on binary image this time)
    bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    # connected components
    cc = cv2.connectedComponents(bin_open)[1]
    # cc=segment_on_dt(bin_open,20)
    return cc


def ali_pipeline(img_green):
    # green channel happends to produce slightly better results
    # than the grayscale image and other channels
    # morphological opening (size tuned on training data)
    circle7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_open = cv2.morphologyEx(img_green, cv2.MORPH_OPEN, circle7)
    # Otsu thresholding
    img_th = cv2.threshold(img_open, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th == 255) > np.sum(img_th == 0)):
        print 'invert ali'
        img_th = cv2.bitwise_not(img_th)
    # second morphological opening (on binary image this time)
    bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    # connected components
    cc = cv2.connectedComponents(bin_open)[1]
    # cc=segment_on_dt(bin_open,20)
    return cc


def ali_pipeline_bk(img_green):
    # green channel happends to produce slightly better results
    # than the grayscale image and other channels
    # morphological opening (size tuned on training data)
    circle7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_open = cv2.morphologyEx(img_green, cv2.MORPH_OPEN, circle7)
    # Otsu thresholding
    img_th = cv2.threshold(img_open, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th == 255) > np.sum(img_th == 0)):
        img_th = cv2.bitwise_not(img_th)
    # second morphological opening (on binary image this time)
    bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    # connected components
    cc = cv2.connectedComponents(bin_open)[1]
    # cc=segment_on_dt(bin_open,20)
    return  # cc
