#!/bin/env python

from __future__ import division
import os
import pandas as pd
import numpy as np
import sys
import logging
from glob import glob

from torch.utils.data import Dataset

from PIL import Image

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split as train_test_split_sk

import cv2

from tqdm import tqdm

from utils import numpy_to_torch

from nuc_trans import binarize, separate_touching_nuclei, erode_mask, redilate_mask, noop_augmentation, nuc_augmentation, affine_augmentation, color_augmentation, preprocess_img, preprocess_mask

class NucleusDataset(Dataset):
    """Dataset for the 2018 Data Science Bowl."""

    @staticmethod
    def read_and_stack(in_img_list):
        return np.sum(np.stack([i*(imread(c_img)>0) for i, c_img in enumerate(in_img_list, 1)], 0), 0)
        #r = (reduce(
        #        np.bitwise_or, [
        #            np.asarray(Image.open(c_img)) for c_img in iter(in_img_list)])).astype(np.uint8)
        #r = r / r.max()
        #return r

    @staticmethod
    def read_image(in_img_list):
        #img = img_as_float(rgb2hsv(rgba2rgb(io.imread(in_img_list[0]))))
        #img = cv2.imread(in_img_list[0])[:,:,1] # as done in ali's pipeline
        #return img, img.shape
        img = Image.open(in_img_list[0])
        return np.array(img.convert('RGB')), img.size

    @staticmethod
    def is_inverted(img, invert_thresh_pd=10.0):
        img_grey = img_as_ubyte(rgb2grey(img))
        img_th = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[1]
        return np.sum(img_th==255)>((invert_thresh_pd/10.0)*np.sum(img_th==0))

    @property
    def dset_type(self):
        return self._dset_type

    @dset_type.setter
    def dset_type(self, value):
        if value not in ('train', 'valid', 'test'):
            raise ValueError('invalid data set type: %s' % value)
        if hasattr(self, '_dset_type') and value == self._dset_type:
            return

        self._dset_type = value
        self.is_preprocessed = False

        if value == 'train':
            self.augment = affine_augmentation()
            self.augment_color = color_augmentation()
        else:
            self.augment = noop_augmentation()
            self.augment_color = noop_augmentation()

        # NOTE on returning more than 2 items:
        # conveniently, the dataloader actually collates dictionaries with arbitrarily many keys,
        # as long as all items with the same key have the same dimensions. Therefore, if we want to
        # test the original images in validation, without resizing, validation batch size has to be 1.
        # as a workaround to get unevenly sized data, we can return the row index to retrieve it in the caller.

        # for debugging, add row index to return_fields. then you can retrieve anything from the row in the caller.

        if self.dset_type == 'train':
            self.return_fields = ['images_prep', 'masks_prep', 'masks_prep_bin', 'num_nuc_inv']
        elif self.dset_type == 'valid':
            self.return_fields = ['images_prep', 'masks', 'masks_bin']
        else:
            self.return_fields = ['images_prep']


    def preprocess(self):

        imgs_prep = []      # preprocessed input images
        masks_bin = []      # binarized masks
        masks_prep = []     # preprocesed masks
        masks_prep_bin = [] # binarized preprocessed masks
        num_nuc_inv = []    # 1 / (#nuclei + 1), used as instance weight

        for i in tqdm(range(len(self.data_df)), desc='prep ' + self.dset_type):

            # training images and masks can be resized, but validation and test cannot
            w = 512 if self.dset_type == 'train' else None

            img = self.data_df['images'].iloc[i]
            imgs_prep.append(preprocess_img(img, w, w))

            if (self.dset_type != 'test'):
                m = self.data_df['masks'].iloc[i]
                prep = preprocess_mask(m, w, w)

                masks_prep.append(prep)
                masks_bin.append(binarize(m))
                masks_prep_bin.append(binarize(prep))
                num_nuc_inv.append(1.0 / (m.flatten().max() + 1.0))

        # for some reason, the following (which is recommended) gives an error:
        # self.data_df.loc[:, 'imgs_prep'] = imgs_prep
        # the following "only" gives a warning:
        self.data_df['images_prep'] = imgs_prep
        if masks_bin:
            self.data_df['masks_bin'] = masks_bin
        if masks_prep:
            self.data_df['masks_prep'] = masks_prep
        if masks_prep_bin:
            self.data_df['masks_prep_bin'] = masks_prep_bin
        if num_nuc_inv:
            self.data_df['num_nuc_inv'] = num_nuc_inv

        self.is_preprocessed = True


    def __init__(
            self,
            root_dir=None,
            stage_name=None,
            group_name=None,
            dset_type='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
        """

        self.root_dir = root_dir
        self.dset_type = dset_type
        self.is_preprocessed = False

        if root_dir is None:
            return

        p =  os.path.join(
            root_dir,
            stage_name +
            '_*',
            '*',
            '*',
            '*')
        all_images = glob(p)
        if len(all_images)==0:
            print "Failed to find any images :( [%s]" % p
            sys.exit(1)
        img_df = pd.DataFrame({'path': all_images})

        def img_id(in_path): return in_path.split('/')[-3]

        def img_type(in_path): return in_path.split('/')[-2]

        def img_group(in_path): return in_path.split('/')[-4].split('_')[1]

        def img_stage(in_path): return in_path.split('/')[-4].split('_')[0]
        img_df['id'] = img_df['path'].map(img_id)
        img_df['type'] = img_df['path'].map(img_type)
        img_df['group'] = img_df['path'].map(img_group)
        img_df['stage'] = img_df['path'].map(img_stage)
        self.img_df = img_df

        data_df = img_df.query('group=="%s"' % group_name)
        data_rows = []
        group_cols = ['stage', 'id']
        for n_group, n_rows in data_df.groupby(group_cols):
            c_row = {
                col_name: col_value for col_name,
                col_value in zip(
                    group_cols,
                    n_group)}
            c_row['images'] = n_rows.query('type == "images"')[
                'path'].values.tolist()
            if self.dset_type != 'test':
                c_row['masks'] = n_rows.query('type == "masks"')[
                    'path'].values.tolist()
            data_rows += [c_row]
        data_df = pd.DataFrame(data_rows)

        logging.debug('reading images')
        ret = data_df['images'].map(self.read_image)
        #(data_df['images'], data_df['format'], data_df['mode'], data_df['size']) = ([x[i] for x in ret] for i in range(4))
        (data_df['images'], data_df['size']) = ([x[i] for x in ret] for i in range(2))

        if self.dset_type != 'test':
            logging.debug('reading masks')
            data_df['masks'] = data_df['masks'].map(
                self.read_and_stack).map(
                    lambda x: x.astype(np.uint8))

        self.data_df = data_df
        logging.debug('done reading data')


    def __len__(self):
        return self.data_df.shape[0]


    def apply_augment(self, cols):
        trans_det = self.augment.to_deterministic()
        trans_det_color = self.augment_color.to_deterministic()

        def trans_if_img(img):
            if isinstance(img, np.ndarray):
                img = trans_det.augment_image(img)
                if len(img.shape) == 3 and img.shape[2] == 3: # exclude the mask from color transformations
                    img = trans_det_color.augment_image(img)
                return numpy_to_torch(trans_det.augment_image(img))
            else:
                # if you want to return something else than an image
                return img

        return dict([(k, trans_if_img(img)) for k, img in cols.items()])


    def __getitem__(self, idx):

        if not self.is_preprocessed:
            raise ValueError('data has not been preprocessed yet')

        row = self.data_df.iloc[idx].to_dict()
        # for debugging:
        # row['idx'] = idx
        return self.apply_augment(dict([(k,row[k]) for k in self.return_fields]))


    def train_test_split(self,  **options):
        """ Return a list of splitted indices from a DataSet.
        Indices can be used with DataLoader to build a train and validation set.

        Arguments:
            A Dataset
            A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
            Shuffling True or False
            Random seed
        """

        df_train, df_valid = train_test_split_sk(self.data_df, **options)
        dset_train = NucleusDataset(dset_type='train')
        dset_train.data_df = df_train

        dset_valid = NucleusDataset(dset_type='valid')
        dset_valid.data_df = df_valid

        return dset_train, dset_valid
