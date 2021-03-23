import subprocess, os, re, time, sys, warnings, glob, shutil
import openslide as slide
from PIL import Image
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
from os import listdir, mkdir, path, makedirs
from os.path import join 
from tqdm import tqdm
import argparse

def thres_saturation(img, t=15, img_channel=3):
    if img_channel==3:
        # typical t = 15
        img = rgb2hsv(img)
        h, w, c = img.shape
        sat_img = img[:, :, 1]
        sat_img = img_as_ubyte(sat_img)
        ave_sat = np.sum(sat_img) / (h * w)
    if img_channel==1:
        # typical t = 20
        h, w = img.shape
        ave_sat = np.sum(img) / (h * w)
    return ave_sat >= t
                        
def slide_to_patch(out_base, img_slides):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        img_class = img_slide.split(path.sep)[-2]
        if img_class == '256':
            img_class = 'input'
            script = 'export_as_tiles_256.groovy'
        if img_class == '512':
            img_class = 'target'
            script = 'export_as_tiles_512.groovy'
        bag_path = join(out_base, img_class, img_name)
        makedirs(bag_path, exist_ok=True)
        image_dir = img_slide
        qupath = os.path.join('QuPath-0.2.3', 'QuPath-0.2.3.exe')
        subprocess.run([qupath, "script", script, "-i", image_dir], shell=True)
        all_patches = glob.glob(os.path.join(img_slide.replace('.tif', ''), '*'))
        for idx, patch in enumerate(all_patches):
            sys.stdout.write('\r Moving files {}/{}'.format(idx, len(all_patches)))
            img = io.imread(patch)
            x_number = patch.split('x=')[1].split(',')[0]
            y_number = patch.split('y=')[1].split(',')[0]
            if img_class == 'input':
                new_x = str(int(int(x_number)/128))
                new_y = str(int(int(y_number)/128))
            if img_class == 'target':
                new_x = str(int(int(x_number)/2/128))
                new_y = str(int(int(y_number)/2/128))
            save_name = os.path.join(bag_path, patch.split(os.sep)[-1]).replace(x_number, new_x).replace(y_number, new_y)
            shutil.move(patch, save_name)
        os.rmdir(img_slide.replace('.tif', ''))
    in_patches = glob.glob(os.path.join(out_base, 'input', '*', '*.png'))
    out_patches = glob.glob(os.path.join(out_base, 'target', '*', '*.png')) 
    for idx in range(len(in_patches)):
        sys.stdout.write('\r Removing background {}/{}'.format(idx, len(in_patches)))
        in_img = io.imread(in_patches[idx])
        if not thres_saturation(in_img, t=17.5, img_channel=1):
            os.remove(in_patches[idx])
            os.remove(out_patches[idx])
                
if __name__ == '__main__':
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser(description='Crop the WSIs into patches')
    args = parser.parse_args()

    print('Cropping patches, this could take a while for big datasets, please be patient')
    path_base = ('WSI/TMA/')
    out_base = ('data/TMA/')
    all_slides = glob.glob(join(path_base, '*/*.tif'))
    slide_to_patch(out_base, all_slides)