import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from dataloaders.dataset_covid import (CovidDataSets, RandomGenerator)
from torch.utils.data import DataLoader
import cv2
import pandas as pd
from utils.distance_metric import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient, compute_robust_hausdorff


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='home/code/SSL/', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str, default='COVID249', help='Name of dataset')
parser.add_argument('--exp', type=str, default='Cross_Pseudo_Supervision', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')


def save_sample_png(png_results_path, file_name , out, rot = False):
    split_list =  file_name.split('_')
    if len(split_list)>2:
        volume_num = split_list[0]+'_'+split_list[1]
        save_img_name = split_list[2]
    else:
        volume_num = split_list[0]
        save_img_name = split_list[1]

    volume_path = os.path.join(png_results_path, volume_num)
    if not os.path.exists(volume_path):
        os.makedirs(volume_path)
    img = out * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    #cv2.imshow('result', th2)
    #cv2.waitKey(0)
    #------------------------------------------------------------------------
    # Save to certain path
    save_img_path = os.path.join(volume_path, save_img_name)
    cv2.imwrite(save_img_path, img)


def pngs_2_niigz(args, png_results_path, nii_results_path, file_volume_name):
    volume_files =  pd.read_excel(args.root_path + "data/{}/{}".format(args.dataset_name, file_volume_name))
    length = volume_files.shape[0]
    for idx in range(length):
        volume_file = volume_files.iloc[idx][0]
        # load original nii
        ori_path= args.root_path + "data/{}/NII/{}".format(args.dataset_name, volume_file+'_ct.nii.gz')
        ori_nii = sitk.ReadImage(ori_path , sitk.sitkUInt8)
        ori_data = sitk.GetArrayFromImage(ori_nii)

        volume_png_folder = os.path.join(png_results_path, volume_file)
        if os.path.exists(volume_png_folder):
            png_files = os.listdir(volume_png_folder)
            for png_file in png_files:
                png_file_slice = int(png_file.split('.')[0])
                png_file_data = cv2.imread(os.path.join(volume_png_folder, png_file), -1)
                ori_data_slice = ori_data[png_file_slice, :,:]
                ori_data_slice = 0*ori_data_slice
                ori_data_slice[png_file_data==255]=1
                ori_data[png_file_slice, :,:] = ori_data_slice

        #save nii
        out_path = os.path.join(nii_results_path, volume_file+'.nii.gz')
        img_new = sitk.GetImageFromArray(ori_data)
        img_new.CopyInformation(ori_nii)
        sitk.WriteImage(img_new, out_path)
        print(volume_file)


def evaluate_nii(args, nii_results_path, file_volume_name):
    nsd_sum = 0
    dice_sum = 0
    hau_sum = 0
    volume_files =  pd.read_excel(args.root_path + "data/{}/{}".format(args.dataset_name, file_volume_name))
    length = volume_files.shape[0]
    for idx in range(length):
        volume_file = volume_files.iloc[idx][0]
        # load gt nii
        gt_path= args.root_path + "data/{}/NII/{}".format(args.dataset_name, volume_file+'_seg.nii.gz')
        gt_nii = nib.load(gt_path)
        gt_data = np.uint8(gt_nii.get_fdata())

        pred_path= nii_results_path + volume_file+'.nii.gz'
        pred_nii = nib.load(pred_path)
        pred_data = np.uint8(pred_nii.get_fdata())
        
        spacing = gt_nii.header.get_zooms()
        
        surface_distances = compute_surface_distances(gt_data, pred_data, spacing_mm=spacing)
        nsd = compute_surface_dice_at_tolerance(surface_distances, 1)        
        dice = compute_dice_coefficient(gt_data, pred_data)
        print(nsd)
        print(dice)
        nsd_sum += nsd
        dice_sum +=dice
    mean_nsd = nsd_sum/length
    mean_dice = dice_sum/length
    
    return mean_nsd, mean_dice     




def get_model_metric(args, model, snapshot_path, model_name, mode='test'):
    model.eval()
    
    file_slice_name = '{}_slice.xlsx'.format(mode)
    file_volume_name = '{}_volume.xlsx'.format(mode)
    val_dataset = CovidDataSets(root_path=args.root_path, dataset_name=args.dataset_name, file_name = file_slice_name)
    print('The overall number of validation images equals to %d' % len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    png_results_path = os.path.join(snapshot_path, '{}_png/'.format(model_name))
    if os.path.isdir(png_results_path) is False:
        os.mkdir(png_results_path)

    for batch_idx, (image, label, file_name, _) in enumerate(val_dataloader):
        image = image.cuda()
        label = label.cuda()

        with torch.no_grad():
            out_main = model(image)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            save_sample_png(png_results_path, file_name = file_name[0], out=out)

    # png results to nii.gz label
    nii_results_path = os.path.join(snapshot_path, '{}_nii/'.format(model_name))
    if os.path.isdir(nii_results_path) is False:
        os.mkdir(nii_results_path)
    pngs_2_niigz(args= args, png_results_path = png_results_path, nii_results_path=nii_results_path, file_volume_name = file_volume_name)
    # evaluate result
    nsd, dice = evaluate_nii(args = args, nii_results_path=nii_results_path, file_volume_name = file_volume_name)
    return nsd, dice


if __name__ == '__main__':
    args = parser.parse_args()
    
