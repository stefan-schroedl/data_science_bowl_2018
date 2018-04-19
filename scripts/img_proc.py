import os
import logging

import math
import numpy as np

import matplotlib.pyplot as plt
#from matplotlib import _cntr as cntr

from torchvision.transforms import ToTensor, ToPILImage

import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from five_crop_aug import FiveCrop

import skimage
from skimage.color import rgb2grey
from skimage import transform, img_as_ubyte, img_as_float, exposure, morphology
from skimage.io import imread
from skimage.feature import peak_local_max
from skimage.util import invert
from skimage.filters import threshold_otsu

from scipy import ndimage as ndi

import torch

from loss import diagnose_errors

from utils import exceptions_str


def is_inverted(img, invert_thresh_pd=10.0):
    img_grey = img_as_ubyte(rgb2grey(img))
    img_th = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)[1]

    return np.sum(
        img_th == 255) > (
        (invert_thresh_pd /
         10.0) *
        np.sum(
            img_th == 0))


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
            p = plt.Polygon([[pt[1], pt[0]] for pt in seg],
                            fill=False, color=color, linewidth=2.0)
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
        # if isinstance(img_conv, torch.Tensor):
        #    sz = img_conv.size()
        #    if len(sz) == 3 and sz[0] == 1:
        #        img_conv = img_conv.squeeze().numpy()
        #    else:
        #        img_conv = img_conv.permute(2,3,1,0).squeeze().numpy()
        axi.imshow(img_conv, cmap='gray')
    if l > max_col:
        for i in range(l, max_col * int(math.ceil(1.0 * l / max_col))):
            c = int(math.floor(1.0 * i / ncol))
            r = i - c * max_col
            ax[c, r].axis('off')
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


# preprocessing

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


# erode masks by one pixel for training, dilate the prediction back at the end

def erode_mask(mask, sz=2):
    struc = morphology.disk(sz)
    return np.where(
        ndi.binary_erosion(
            mask,
            structure=struc,
            border_value=1),
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

def preprocess_img(x, dset_type):
    if dset_type == 'train':
        w = 512
        h = 512
        x = transform.resize(x, (w, h))
        # transform.resize() changes type to float!
        x = img_as_ubyte(x)
    return x


def preprocess_mask(x, dset_type):
    if dset_type == 'train':
        w = 512
        h = 512
        x = transform.resize(x, (w, h))
        # transform.resize() changes type to float!
        x = img_as_ubyte(x)
        x = erode_mask(x)
    return x


def postprocess_prediction(pred, sz=2, max_clusters_for_dilation=100, thresh=0.0):
    # input is torch tensor of model prediction
    # output is in numpy format
    if not isinstance(pred, np.ndarray):
        pred_np = pred.data.cpu().numpy().squeeze()
    img_th = (pred_np > thresh).astype(int)

    img_th = redilate_mask(img_th, sz=sz, skip_clusters=max_clusters_for_dilation)
    return img_th, pred_np


def noop_augmentation():
    return iaa.Sequential([iaa.Noop()])


def nuc_augmentation():
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf(iaa.Affine(rotate=(-15, 15),
                             mode=ia.ALL,
                             order=[0, 1]),
                  iaa.Affine(shear=(-15, 15),
                             mode=ia.ALL,
                             order=[0, 1])),
        iaa.OneOf(iaa.Scale((0.5, 1.0)),
                  iaa.Crop(percent=(0.0, 0.25), keep_size=False))])


def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    contours = skimage.measure.find_contours(img, 0) # outside contour
    for contour in contours:
        contour = contour.astype(np.uint8)
        img_contour[contour[:,0], contour[:,1]] = 1
    return img_contour

#def get_contour(img):
#    img_contour = np.zeros_like(img).astype(np.uint8)
#    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 4)
#    return img_contour


########
# from https://github.com/neptune-ml/open-solution-data-science-bowl-2018/wiki

def affine_augmentation(size):
    return iaa.Sequential([
        # General
        iaa.SomeOf((1, 2),
                   [iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.Affine(rotate=(-45, 45), mode='symmetric', order=[0, 1]),
                    iaa.Affine(scale=(.8, 1.2), mode='symmetric', order=[0, 1])
                    #iaa.CropAndPad(percent=(-0.25, 0.25), pad_cval=0)
                    ]),
        # Deformations
        # WARNING: PiecewiseAffine basically erases contour lines!!!
        # iaa.PiecewiseAffine(scale=(0.00, 0.06))
        FiveCrop(size)
    ])


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

########


def random_rotate90_transform2(*images):
    k = np.random.randint(4)
    return [random_rotate90_transform1(img, k) for img in images]


def random_rotate90_transform1(image, k):
    if k == 0:
        pass
    elif k == 1:
        image = image.transpose(1, 0, 2)
        image = np.flip(image, 0).copy()
        if k == 2:
            image = np.flip(np.flip(image, 0), 1).copy()
        elif k == 3:
            image = image.transpose(1, 0, 2)
            image = np.flip(image, 1).copy()
        return image

####

# original classic processing, 'ali's pipeline'


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
