import os
import logging

import math
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

import skimage
import skimage.color
import skimage.morphology
from scipy import ndimage as ndi

# preprocessing
def as_segmentation(img):
        return (img>0).astype(np.uint8)

def separate_touching_nuclei(labeled_mask, sz=2):
    struc = skimage.morphology.disk(sz)

    img_sum = np.zeros(labeled_mask.shape)
    for j in range(1, labeled_mask.max()+1):
        m = (labeled_mask == j).astype(np.uint8)
        m = skimage.morphology.binary_dilation(m, struc)
        img_sum += m
    ov = np.maximum(0, img_sum - 1)

    mask_corrected = np.where(ov == 0, labeled_mask, 0)
    return mask_corrected, ov

# erode masks by one pixel for training, dilate the prediction back at the end

def erode_mask(mask, sz=2):
        struc = skimage.morphology.disk(sz)
        return np.where(ndi.binary_erosion(mask, structure=struc, border_value=1), mask, 0)

def redilate_mask(mask_seg, sz=2, skip_clusters=1e20):
        mask_l, n = ndi.label(mask_seg)
        if n >= skip_clusters:
                # too slow, use shortcut
                return mask_l
        struc = skimage.morphology.disk(sz)
        mask_dil = np.zeros(mask_l.shape, dtype=np.int32)

        for cluster in range(1, n + 1):
                cur = (mask_l == cluster)
                cur = skimage.morphology.binary_dilation(cur, struc)
                mask_dil = np.where(cur, cluster, mask_dil)
        return mask_dil

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
                iaa.OneOf(iaa.Scale((0.5,1.0)),
                          iaa.Crop(percent=(0.0,0.25), keep_size=False))])


def random_rotate90_transform2(*images):
    k = np.random.randint(4)
    return [random_rotate90_transform1(img, k) for img in images]

def random_rotate90_transform1(image, k):
        if k == 0:
            pass
        elif k == 1:
            image = image.transpose(1,0,2)
            image = np.flip(image,0).copy()
        if k == 2:
            image = np.flip(np.flip(image,0),1).copy()
        elif k == 3:
            image = image.transpose(1,0,2)
            image = np.flip(image,1).copy()
        return image
