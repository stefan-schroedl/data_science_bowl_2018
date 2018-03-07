#%pycat dataset.py

from __future__ import division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from PIL import Image
from torch.autograd import Variable
import random
import math
from functools import reduce

from glob import glob

import skimage
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte


class NucleusDataset(Dataset):
    """Nucleus dataset."""

    @staticmethod
    def read_and_stack(in_img_list):
        # return np.sum(np.stack([i*(imread(c_img)>0) for i, c_img in
        # enumerate(in_img_list, 1)], 0), 0)
        r = img_as_float(
            reduce(
                np.bitwise_or, [
                    imread(c_img) for c_img in iter(in_img_list)]))
        r = r / r.max()
        return r

    @staticmethod
    def read_image(in_img_list):
        #img = img_as_float(rgb2hsv(rgba2rgb(io.imread(in_img_list[0]))))
        # return torch.from_numpy(img)
        img = Image.open(in_img_list[0])
        img = img.convert('RGB')
        return img

    def __init__(
            self,
            root_dir,
            stage_name='stage1',
            group_name='train',
            transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        # print os.path.join(dsb_data_dir, stage_name + '_*', '*', '*', '*')
        all_images = glob(
            os.path.join(
                root_dir,
                stage_name +
                '_*',
                '*',
                '*',
                '*'))

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
        data_df['images'] = data_df['images'].map(self.read_image)
        data_df['masks'] = data_df['masks'].map(
            self.read_and_stack).map(
            lambda x: x.astype(int))

        # print data_df.describe()
        self.data_df = data_df

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):

        sample = self.data_df["images"].iloc[idx]
        masks = torch.from_numpy(
            self.data_df["masks"].iloc[idx]).type(
            torch.FloatTensor)
        # insert 1 dummy channel, to be consistent with output
        masks = torch.unsqueeze(masks, 0)
        if self.transform:
            sample = self.transform(sample)

        return sample, masks

# https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4


def train_valid_split(dataset, test_size=0.25, shuffle=False, random_seed=0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.

    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = len(dataset)
    indices = list(range(1, length))

    if shuffle:
        random.seed(random_seed)
        random.shuffle(indices)

    if isinstance(test_size, float):
        split = int(math.floor(test_size * length + 0.5))
    elif isinstance(test_size, int):
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]
