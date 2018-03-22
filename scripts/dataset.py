#%pycat dataset.py

from __future__ import division
import os
import torch
import pandas as pd
import numpy as np
import sys
import random
import math
from functools import reduce

from glob import glob

from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image

import skimage
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2grey

import sklearn
from sklearn.model_selection import train_test_split as train_test_split_sk

import cv2

import tqdm

from nuc_trans import as_segmentation, segment_separate_touching_nuclei

class NucleusDataset(Dataset):
    """Nucleus dataset."""

    @staticmethod
    def read_and_stack(in_img_list):
        return np.sum(np.stack([i*(imread(c_img)>0) for i, c_img in enumerate(in_img_list, 1)], 0), 0)
        #r = (reduce(
        #        np.bitwise_or, [
        #            np.asarray(Image.open(c_img)) for c_img in iter(in_img_list)])).astype(np.uint8)
        #r = r / r.max()
        return r

    @staticmethod
    def read_image(in_img_list):
        #img = img_as_float(rgb2hsv(rgba2rgb(io.imread(in_img_list[0]))))
        img = Image.open(in_img_list[0])
        return np.array(img.convert('RGB')), img.size

    @staticmethod
    def is_inverted(img, invert_thresh_pd=10.0):
        img_grey = img_as_ubyte(rgb2grey(img))
        img_th = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[1]
        return np.sum(img_th==255)>((invert_thresh_pd/10.0)*np.sum(img_th==0))

    def __init__(
            self,
            root_dir=None,
            stage_name='stage1',
            group_name='train',
            preprocess=segment_separate_touching_nuclei,
            transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        if root_dir is None:
            return

        all_images = glob(
            os.path.join(
                root_dir,
                stage_name +
                '_*',
                '*',
                '*',
                '*'))
        if len(all_images)==0:
            print "Failed to find any images :("
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
            c_row['masks'] = n_rows.query('type == "masks"')[
                'path'].values.tolist()
            c_row['images'] = n_rows.query('type == "images"')[
                'path'].values.tolist()
            data_rows += [c_row]

        data_df = pd.DataFrame(data_rows)

        ret = data_df['images'].map(self.read_image)
        #(data_df['images'], data_df['format'], data_df['mode'], data_df['size']) = ([x[i] for x in ret] for i in range(4))
        (data_df['images'], data_df['size']) = ([x[i] for x in ret] for i in range(2))

        if np.any([len(x) > 0 for x in data_df['masks']]):
            data_df['masks'] = data_df['masks'].map(
                self.read_and_stack).map(
                    lambda x: x.astype(np.uint8))
            data_df['masks_seg'] = data_df['masks'].map(preprocess)
        else:
            data_df['masks'] = data_df['images'].map(lambda x: np.zeros(x.shape))
            data_df['masks_seg'] = data_df['masks']

        data_df['inv'] = data_df['images'].map(self.is_inverted)

        self.data_df = data_df


    def __len__(self):
        return self.data_df.shape[0]


    def __getitem__(self, idx):

        sample = self.data_df["images"].iloc[idx]
        masks =  self.data_df["masks"].iloc[idx]
        masks_seg =  self.data_df["masks_seg"].iloc[idx]
        if self.transform:
            sample, masks, masks_seg = self.transform(sample, masks, masks_seg)
        return sample, (masks, masks_seg)


    def train_test_split(self, **options):
        """ Return a list of splitted indices from a DataSet.
        Indices can be used with DataLoader to build a train and validation set.

        Arguments:
            A Dataset
            A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
            Shuffling True or False
            Random seed
        """

        df_train, df_test = train_test_split_sk(self.data_df, **options)
        dset_train = NucleusDataset(transform=self.transform)
        dset_train.data_df = df_train

        dset_test = NucleusDataset(transform=self.transform)
        dset_test.data_df = df_test

        return dset_train, dset_test
