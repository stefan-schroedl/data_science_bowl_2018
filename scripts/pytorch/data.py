#!/usr/bin/env python

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from glob import glob
import skimage
from skimage.io import imread
import matplotlib.pyplot as plt

class NucleusDataset(Dataset):
    """Nucleus dataset."""

    @staticmethod
    def read_and_stack(in_img_list):
        return np.sum(np.stack([i*(imread(c_img)>0) for i, c_img in enumerate(in_img_list, 1)], 0), 0)

    @staticmethod
    def read_image(in_img_list):
        return torch.from_numpy(io.imread(in_img_list[0]))

    def __init__(self, root_dir, stage_name='stage1', group_name='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        all_images = glob(os.path.join(dsb_data_dir, stage_name + '_*', '*', '*', '*'))
      
        img_df = pd.DataFrame({'path': all_images})
        img_id = lambda in_path: in_path.split('/')[-3]
        img_type = lambda in_path: in_path.split('/')[-2]
        img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
        img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
        img_df['id'] = img_df['path'].map(img_id)
        img_df['type'] = img_df['path'].map(img_type)
        img_df['group'] = img_df['path'].map(img_group)
        img_df['stage'] = img_df['path'].map(img_stage)
        self.img_df = img_df

        data_df = img_df.query('group=="%s"' % group_name)
        data_rows = []
        group_cols = ['stage', 'id']
        for n_group, n_rows in data_df.groupby(group_cols):
            c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
            c_row['masks'] = n_rows.query('type == "masks"')['path'].values.tolist()
            c_row['images'] = n_rows.query('type == "images"')['path'].values.tolist()
            data_rows += [c_row]
       
        data_df = pd.DataFrame(data_rows)
        data_df['images'] = data_df['images'].map(self.read_image)
        data_df['masks'] = data_df['masks'].map(self.read_and_stack).map(lambda x: x.astype(int))
        
        self.data_df = data_df

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        
        sample = self.data_df["images"].iloc[idx]
        masks = self.data_df["masks"].iloc[idx]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, masks
