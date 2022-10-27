import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from itertools import cycle
import numpy as np
import cv2 

from dataloaders import utils
from dataloaders.dataset_covid import (CovidDataSets, RandomGenerator)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from test_covid import get_model_metric
from models.pix2pix_model import Pix2PixModel, get_opt


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/code/SSL/', help='Name of Experiment')

parser.add_argument('--labeled_per', type=float, default=0.1, help='percent of labeled data')
if False:
    parser.add_argument('--dataset_name', type=str, default='COVID249', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_0.1_l.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_0.1_u.xlsx', help='Name of dataset')
else:
    parser.add_argument('--dataset_name', type=str, default='MOS1000', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_slice_label.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_slice_unlabel.xlsx', help='Name of dataset')

parser.add_argument('--exp', type=str, default='Template', help='experiment_name')

parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_epoch', type=int, default=20, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')

# label and unlabel
parser.add_argument('--batch_size_label', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--batch_size_unlabel', type=int, default=8, help='batch_size per gpu')

args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_epoch = args.max_epoch
    excel_file_name_label = args.excel_file_name_label
    excel_file_name_unlabel = args.excel_file_name_unlabel


    # create model
    teacher_model = net_factory(net_type=args.model)
    student_model = net_factory(net_type=args.model)
    teacher_model.load_state_dict(torch.load(args.teacher_path))
    teacher_model.eval()
    opt = get_opt()
    syn_model = Pix2PixModel(opt)
    syn_model.eval()
    
    # Define the dataset
    labeled_train_dataset = CovidDataSets(root_path=args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_label, aug = True)
    unlabeled_train_dataset = CovidDataSets(root_path=args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_unlabel, aug = True)
    print('The overall number of labeled training image equals to %d' % len(labeled_train_dataset))
    print('The overall number of unlabeled training images equals to %d' % len(unlabeled_train_dataset))

    student_model.train()

    optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')

    # Define the dataloader
    labeled_dataloader = DataLoader(labeled_train_dataset, batch_size = args.batch_size_label, shuffle = True, num_workers = 4, pin_memory = True)
    unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size = args.batch_size_unlabel, shuffle = True, num_workers = 4, pin_memory = True)
    
    iter_num_s = 0
    max_iterations_s = max_epoch * len(unlabeled_dataloader)
    for epoch in range(max_epoch):
        print("Start epoch ", epoch+1, "!")

        tbar = tqdm(range(len(unlabeled_dataloader)), ncols=70)
        labeled_dataloader_iter = iter(labeled_dataloader)
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for batch_idx in tbar:

            try:
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()
            except StopIteration:
                labeled_dataloader_iter = iter(labeled_dataloader)
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()

                style_output_global_positive_list =[]
        
            input_ul, target_ul, file_name_ul , lung_ul = unlabeled_dataloader_iter.next()
            input_ul, target_ul, lung_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True), lung_ul.cuda(non_blocking=True)
            input_l, target_l, lung_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True), lung_l.cuda(non_blocking=True)


            # Add new impelmentation here: design the training process
            #--------------------------------------------------------------------------------------------------------
            #********************************************************************************************************
            #--------------------------------------------------------------------------------------------------------


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_s.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
    writer.close()



if __name__ == "__main__":
    # for reproducing
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    seed = 66
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    snapshot_path = "{}exp/{}/exp_{}_{}_{}".format(args.root_path, args.dataset_name, args.exp, args.labeled_per, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)