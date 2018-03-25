import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import imgaug
from imgaug import augmenters as iaa


# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel


# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import matplotlib.pyplot as plt

import skimage
import skimage.color
import skimage.morphology
from scipy import ndimage as ndi

# preprocessing
def as_segmentation(img):
        return (img>0).astype(int)

def separate_touching_nuclei(labeled_mask, sz=2):
    struc = skimage.morphology.disk(sz)

    img_sum = np.zeros(labeled_mask.shape)
    for j in range(1, labeled_mask.max()+1):
        m = (labeled_mask == j).astype(int)
        m = skimage.morphology.binary_dilation(m, struc)
        img_sum += m
    ov = np.maximum(0, img_sum - 1)

    mask_corrected = np.where(ov == 0, labeled_mask, 0)
    return mask_corrected, ov

# erode masks by one pixel for training, dilate the prediction back at the end

def erode_mask(mask, sz=2):
        struc = skimage.morphology.disk(sz)
        return np.where(ndi.binary_erosion(mask, structure=struc, border_value=1), mask, 0)

def redilate_mask(mask_seg, sz=2):
        struc = skimage.morphology.disk(sz)
        mask_l, n = ndi.label(mask_seg)
        mask_dil = np.zeros(mask_l.shape, dtype=np.int32)
        for cluster in range(1, n + 1):
                cur = (mask_l == cluster).astype(int)
                cur = skimage.morphology.binary_dilation(cur, struc)
                mask_dil[cur > 0.5] = cluster
        return mask_dil


# kaggle science bowl-2 : ################################################################

def resize_to_factor2(image, mask, factor=16):

    H,W = image.shape[:2]
    h = (H//factor)*factor
    w = (W //factor)*factor
    return fix_resize_transform2(image, mask, w, h)



def fix_resize_transform2(image, mask, w, h):
    H,W = image.shape[:2]
    if (H,W) != (h,w):
        image = cv2.resize(image,(w,h))

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask,(w,h),cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)
    return image, mask




def fix_crop_transform2(image, mask, x,y,w,h):

    H,W = image.shape[:2]
    assert(H>=h)
    assert(W >=w)

    if (x==-1 & y==-1):
        x=(W-w)//2
        y=(H-h)//2

    if (x,y,w,h) != (0,0,W,H):
        image = image[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    return image, mask


def random_crop_transform2(image, mask, w,h):
    H,W = image.shape[:2]

    if H!=h:
        y = np.random.choice(H-h)
    else:
        y=0

    if W!=w:
        x = np.random.choice(W-w)
    else:
        x=0

    return fix_crop_transform2(image, mask, x,y,w,h)



def random_horizontal_flip_transform2(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img) ##left-right
        mask  = cv2.flip(mask,1)
    return image, mask

def random_vertical_flip_transform2(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,0)
        mask  = cv2.flip(mask,0)
    return image, mask


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


def random_shift_scale_rotate_transform2( image, mask,
                        shift_limit=[-0.0625,0.0625], scale_limit=[1/1.2,1.2],
                        rotate_limit=[-15,15], borderMode=cv2.BORDER_REFLECT_101 , u=0.5):

    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height, width, channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        sx    = scale
        sy    = scale
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        mask = mask.astype(np.float32)
        mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = mask.astype(np.int32)

    return image, mask




# single image ########################################################

#agumentation (photometric) ----------------------
def random_brightness_shift_transform(image, limit=[16,64], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        image = image + alpha*255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_brightness_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        image = alpha*image
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_contrast_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha*image  + gray
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_saturation_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        coef  = np.array([[[0.114, 0.587,  0.299]]])
        gray  = image * coef
        gray  = np.sum(gray,axis=2, keepdims=True)
        image = alpha*image  + (1.0 - alpha)*gray
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def random_hue_transform(image, limit=[-0.1,0.1], u=0.5):
    if random.random() < u:
        h = int(np.random.uniform(limit[0], limit[1])*180)
        #print(h)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def random_noise_transform(image, limit=[0, 0.5], u=0.5):
    if random.random() < u:
        H,W = image.shape[:2]
        noise = np.random.uniform(limit[0],limit[1],size=(H,W))*255

        image = image + noise[:,:,np.newaxis]*np.array([1,1,1])
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# geometric ---
def resize_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = (height//factor)*factor
    w = (width //factor)*factor
    return fix_resize_transform(image, w, h)


def fix_resize_transform(image, w, h):
    height,width = image.shape[:2]
    if (height,width) != (h,w):
        image = cv2.resize(image,(w,h))
    return image



def pad_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = math.ceil(height/factor)*factor
    w = math.ceil(width/factor)*factor

    image = cv2.copyMakeBorder(image, top=0, bottom=h-height, left=0, right=w-width,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    return image


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    print('\nsucess!')
