from __future__ import print_function, division
import os, glob, random
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color, img_as_ubyte, exposure
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

class Pair_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):   
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):             
        img = io.imread(self.files_list.iloc[idx, 0])
        if img.shape[0] == 512 or img.shape[1] == 512:
            img = exposure.rescale_intensity(img_as_float(img), in_range=(0.1, 0.9), out_range=(0, 1))
        else:
            img = exposure.rescale_intensity(img_as_float(img), in_range=(0.2, 0.8), out_range=(0, 1))
        sample = {'input': img, 'output': img}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string 
    
class Drop(object):
    def __init__(self, drop_mask=None):
        self.output_size = drop_mask

    def __call__(self, sample):
        in_img = sample['input']
        out_img = np.multiply(in_img, drop_mask)
        return {'input': in_img, 'output': out_img}

class ToTensor(object):
    def __call__(self, sample):
        in_img, out_img = sample['input'], sample['output']
        return {'input': transforms.functional.to_tensor(np.array(in_img)), 'output': transforms.functional.to_tensor(np.array(out_img))}

def show_patch(dataloader, index = 0, img_channel=1):
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == index:         
            input_batch, output_batch = sample_batched['input'], sample_batched['output']
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
            plt.figure(figsize=(20, 10))
            grid = utils.make_grid(input_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bicubic')  
            plt.axis('off')
            plt.figure(figsize=(20, 10))
            grid = utils.make_grid(output_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bicubic')
            plt.axis('off')
            break
            
def generate_compress_csv(resolution=256):
    imgs = glob.glob(os.path.join('data', str(resolution), 'noisy', '*.png'))
    imgs_df = pd.DataFrame(imgs)
    imgs_df = imgs_df.sample(frac=1).reset_index(drop=True)
    train_df = pd.DataFrame(imgs_df[0:int(0.8*len(imgs_df))])
    valid_df = pd.DataFrame(imgs_df[int(0.8*len(imgs_df)):])
    train_df.to_csv(os.path.join('data', str(resolution), 'single-self-train.csv'), index=False)
    valid_df.to_csv(os.path.join('data', str(resolution), 'single-self-valid.csv'), index=False)
    
def compress_csv_path(csv='train', resolution=256):
    if csv =='train':
        return os.path.join('data', str(resolution), 'single-self-train.csv')
    if csv =='valid':
        return os.path.join('data', str(resolution), 'single-self-valid.csv')