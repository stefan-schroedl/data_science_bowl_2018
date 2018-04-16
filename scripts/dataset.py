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

from img_proc import numpy_img_to_torch, binarize, noop_augmentation, nuc_augmentation, affine_augmentation, color_augmentation, preprocess_img, preprocess_mask, get_contour


class NucleusDataset(Dataset):

    """
    Dataset for the 2018 Data Science Bowl.

    usage pattern:
    ds = NucleusDataset(...)
    train_ds, valid_ds = ds.train_test_split(...)
    train_ds.preprocess() # depends on dset_type
    train_ds.return_fields = (...)
    valid_ds ...
    ...
    train_loader = DataLoader(train_ds, ...)
    ...
    for it, row in enumerate(train_loader):

    """


    @staticmethod
    def read_and_stack(in_img_list):
        return np.sum(np.stack([i * (imread(c_img) > 0)
                                for i, c_img in enumerate(in_img_list, 1)], 0), 0)
        # r = (reduce(
        #        np.bitwise_or, [
        #            np.asarray(Image.open(c_img)) for c_img in iter(in_img_list)])).astype(np.uint8)
        #r = r / r.max()
        # return r


    @staticmethod
    def read_image(in_img_list):
        print "READ IMG",in_img_list[0]
        #img = img_as_float(rgb2hsv(rgba2rgb(io.imread(in_img_list[0]))))
        # img = cv2.imread(in_img_list[0])[:,:,1] # as done in ali's pipeline
        # return img, img.shape
        img = Image.open(in_img_list[0])
        return np.array(img.convert('RGB')), img.size


    @property
    def dset_type(self):
        """The dataset type determines preprocessing and augmentation"""
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


    def __init__(self, root_dir=None, stage_name=None, group_name=None, dset_type='train'):
        """
        Read all images and masks into memory.

        directory structure is assumed to be:
            <root>/<stage>_<train/test>/{images,masks}/*png

        args:
            root_dir (string): directory with all the images.
            stage_name (string): stage of data
            group_name (string): typically 'train' or 'test'. No masks are available for 'test'.
            dset_type: dataset mode, either 'train', 'valid', or 'test'.
        """

        self.root_dir = root_dir
        self.dset_type = dset_type
        self.is_preprocessed = False

        if root_dir is None:
            return

        p = os.path.join(root_dir, stage_name + '_*', '*', '*', '*')
        all_images = glob(p)
        if len(all_images) == 0:
            raise ValueError("Failed to find any images :( [%s]" % p)
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
        (data_df['images'], data_df['size']) = (
            [x[i] for x in ret] for i in range(2))

        if self.dset_type != 'test':
            logging.debug('reading masks')
            data_df['masks'] = data_df['masks'].map(
                self.read_and_stack).map(
                    lambda x: x.astype(np.uint8))

        self.data_df = data_df

        logging.debug('done reading data')


    def preprocess(self):

        imgs_prep = []      # preprocessed input images
        masks_bin = []      # binarized masks
        masks_prep = []     # preprocesed masks
        masks_prep_bin = [] # binarized preprocessed masks
        contours = []       # contours on the eroded mask for multi-task
        inst_wt = []        # instance weights, 1 / (#nuclei + 1), used as instance weight

        for i in tqdm(range(len(self.data_df)), desc='prep ' + self.dset_type):

            img = self.data_df['images'].iloc[i]
            imgs_prep.append(preprocess_img(img, self.dset_type))

            if (self.dset_type != 'test'):
                m = self.data_df['masks'].iloc[i]
                prep = preprocess_mask(m, self.dset_type)

                masks_prep.append(prep)
                masks_bin.append(binarize(m))
                prep_bin = binarize(prep)
                masks_prep_bin.append(prep_bin)
                contours.append(get_contour(prep_bin))
                w = (1.0 / (m.flatten().max() + 1.0)).astype(np.float32)
                inst_wt.append(w)

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
        if contours:
            self.data_df['contours'] = contours

        if inst_wt:
            # normalize!
            inst_wt = inst_wt / np.mean(inst_wt)
            self.data_df['inst_wt'] = inst_wt

        self.is_preprocessed = True


    def apply_augment(self, cols):
        trans_det = self.augment.to_deterministic()
        trans_det_color = self.augment_color.to_deterministic()

        def trans_if_img(img):
            return numpy_img_to_torch(img)
            if isinstance(img, np.ndarray):
                img = trans_det.augment_image(img)
                if len(
                        img.shape) == 3 and img.shape[2] == 3:  # exclude the mask from color transformations
                    img = trans_det_color.augment_image(img)
                return numpy_img_to_torch(trans_det.augment_image(img))
            else:
                # if you want to return something else than an image
                return img

        return dict([(k, trans_if_img(img)) for k, img in cols.items()])


    def __len__(self):
        return self.data_df.shape[0]


    def __getitem__(self, idx):

        # NOTE on returning more than 2 items (train and test image):
        # conveniently, the dataloader actually collates dictionaries with arbitrarily many keys,
        # as long as all items with the same key have the same dimensions. Therefore, if we want to
        # test the original images in validation, without resizing, validation batch size has to be 1.
        # as a workaround to get unevenly sized data, we can return the row
        # index to retrieve it in the caller.

        # for debugging, add row index to return_fields. then you can retrieve
        # anything from the row in the caller.

        if not self.is_preprocessed:
            raise ValueError('data has not been preprocessed yet')

        if not hasattr(self, 'return_fields'):
            raise ValueError('return_fields has to be set before calling __getitem__')

        row = self.data_df.iloc[idx].to_dict()
        return self.apply_augment(
            dict([(k, row[k]) for k in self.return_fields]))


    def train_test_split(self, **options):
        """ Return splitted train and validation datasets.
        options are passed to sklearn.model_selection.train_test_split, see there.
        """

        if 'stratify' in options:
            if options['stratify'] == False:
                del options['stratify']
            else:
                # stratify by image size.
                # NOTE: image sizes that occur only once will not be stratifiable, remove
                grouped = self.data_df.groupby('size', sort=False)
                to_delete = []
                for group in grouped:
                    if len(group[1]) <= 1:
                        to_delete.append(group[0])
                for sz in to_delete:
                    logging.warn('img size {} only occurs once, deleting due to stratification'.format(sz))
                    self.data_df = self.data_df[self.data_df['size'] != sz]

                options['stratify'] = self.data_df['images'].map(lambda x: '{}'.format(x.size))

        df_train, df_valid = train_test_split_sk(self.data_df, **options)
        dset_train = NucleusDataset(dset_type='train')
        dset_train.data_df = df_train

        dset_valid = NucleusDataset(dset_type='valid')
        dset_valid.data_df = df_valid

        return dset_train, dset_valid
