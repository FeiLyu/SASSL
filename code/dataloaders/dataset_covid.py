import os
import os.path
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import pandas as pd
from PIL import Image


class CovidDataSets(Dataset):
    def __init__(self, root_path=None, dataset_name='COVID249', file_name = 'val_slice.xlsx', aug=False):
        self.root_path = root_path
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.file_path = root_path + "data/{}/{}".format(dataset_name, file_name)
        self.aug = aug


        excelData = pd.read_excel(self.file_path)
        length = excelData.shape[0] 
        self.paths = []
        for i in range(length): 
            file_name_i =  excelData.iloc[i][0]
            self.paths.append(file_name_i)

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        case = self.paths[idx]

        case_img_path = self.root_path + "data/{}/PNG/images/{}".format(self.dataset_name,  case) 
        case_label_path = self.root_path + "data/{}/PNG/labels/{}".format(self.dataset_name, case) 
        case_lung_path = self.root_path + "data/{}/PNG/lung/{}".format(self.dataset_name, case) 
        
        image = Image.open(case_img_path)

        if os.path.exists(case_label_path):
            label = Image.open(case_label_path)
        else:
            label = Image.open(case_lung_path)
        
        lung = Image.open(case_lung_path)

        if self.aug:
            if random.random() > 0.5:
                image, label, lung = random_rot_flip(image, label, lung)
            elif random.random() > 0.5:
                image, label, lung = random_rotate(image, label, lung)

        image = (torch.from_numpy(np.asarray(image).astype(np.float32)).permute(2, 0, 1).contiguous())/255.0
        label = torch.from_numpy(np.asarray(label).astype(np.uint8))
        lung = torch.from_numpy(np.asarray(lung).astype(np.uint8))

        return image, label, case, lung


def random_rot_flip(image, label, lung):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    lung = np.rot90(lung, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    lung = np.flip(lung, axis=axis).copy()
    return image, label, lung


def random_rotate(image, label, lung):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    lung = ndimage.rotate(lung, angle, order=0, reshape=False)
    return image, label, lung


def rotate_90(image, label, lung):
    angle = 90
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    lung = ndimage.rotate(lung, angle, order=0, reshape=False)
    return image, label, lung

def rotate_n90(image, label, lung):
    angle = -90
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    lung = ndimage.rotate(lung, angle, order=0, reshape=False)
    return image, label, lung



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, label):
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        return image, label


