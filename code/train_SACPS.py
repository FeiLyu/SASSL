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

from dataloaders import utils
from dataloaders.dataset_covid import (CovidDataSets, RandomGenerator)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from test_covid import get_model_metric
import cv2
from models.pix2pix_model import Pix2PixModel, get_opt

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/', help='Name of Experiment')

parser.add_argument('--consistency_syn', type=float, default=0.5, help='consistency')
parser.add_argument('--consistency_pseudo', type=float, default=0.5, help='consistency')

parser.add_argument('--labeled_per', type=float, default=0.1, help='percent of labeled data')
if True:
    parser.add_argument('--dataset_name', type=str, default='COVID249', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_0.1_l.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_0.1_u.xlsx', help='Name of dataset')
else:
    parser.add_argument('--dataset_name', type=str, default='MOS1000', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_slice_label.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_slice_unlabel.xlsx', help='Name of dataset')

parser.add_argument('--exp', type=str, default='sacps', help='experiment_name')
parser.add_argument('--model', type=str, default='unet2', help='model_name')
parser.add_argument('--max_epoch', type=int, default=20, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--batch_size_label', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--batch_size_unlabel', type=int, default=8, help='batch_size per gpu')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=10.0, help='consistency_rampup')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    excel_file_name_label = args.excel_file_name_label
    excel_file_name_unlabel = args.excel_file_name_unlabel


    # create model
    model1 = net_factory(net_type=args.model)
    model2 = net_factory(net_type=args.model)
    opt = get_opt()
    syn_model = Pix2PixModel(opt)
    syn_model.eval()
    
    # Define the dataset
    labeled_train_dataset = CovidDataSets(root_path=args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_label, aug = True)
    unlabeled_train_dataset = CovidDataSets(root_path=args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_unlabel, aug = True)
    print('The overall number of labeled training image equals to %d' % len(labeled_train_dataset))
    print('The overall number of unlabeled training images equals to %d' % len(unlabeled_train_dataset))


    # start training
    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    #logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance1 = 0.0
    best_performance2 = 0.0

    # Define the dataloader
    labeled_dataloader = DataLoader(labeled_train_dataset, batch_size = args.batch_size_label, shuffle = True, num_workers = 4, pin_memory = True)
    unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size = args.batch_size_unlabel, shuffle = True, num_workers = 4, pin_memory = True)
    max_iterations = max_epoch * len(unlabeled_dataloader)

    for epoch in range(max_epoch):
        print("Start epoch ", epoch, "!")

        style_output_global_positive_list =[]
        style_output_global_list =[]
        
        tbar = tqdm(range(len(unlabeled_dataloader)), ncols=70)
        labeled_dataloader_iter = iter(labeled_dataloader)
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for batch_idx in tbar:
            try:
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()
            except StopIteration:
                labeled_dataloader_iter = iter(labeled_dataloader)
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()
                print('length: style_output_global_positive_list')
                print(len(style_output_global_positive_list))

                style_output_global_positive_list =[]
                style_output_global_list =[]
        
            input_ul, target_ul, file_name_ul , lung_ul = unlabeled_dataloader_iter.next()
            input_ul, target_ul, lung_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True), lung_ul.cuda(non_blocking=True)
            input_l, target_l, lung_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True), lung_l.cuda(non_blocking=True)

            if input_l.shape[0]!=args.batch_size_label:
                continue

            # generate style codes from labeld data
            lung_l[target_l>0] = 3
            style_output_list =[]
            len_style = input_l.shape[0]
            for style_idx in range(len_style):
                with torch.no_grad():
                    normalize_input = transforms.functional.normalize(input_l[style_idx], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    output_ul_style = syn_model(normalize_input.unsqueeze_(0), lung_l[style_idx:style_idx+1].unsqueeze_(1), mode='style', data_path=[file_name_l[style_idx]])
                    output_ul_style_check = torch.mean(output_ul_style, 2)
                    # label 4 does not exist
                    if output_ul_style_check[0,3] != 0:
                        style_output_global_positive_list.append((output_ul_style).detach())
                    style_output_list.append((output_ul_style).detach())
            
            style_output = style_output_global_positive_list #+ style_output_list
            if len(style_output)>10:
                style_output = random.choices(style_output, k=10)

            # get pseudo labels from model1 for unlabeled data
            model1.eval()
            with torch.no_grad():
                outputs_unlabeled  = model1(input_ul)
                outputs_unlabeled_soft = torch.softmax(outputs_unlabeled, dim=1)
                pseudo_labels_s1 = torch.argmax(outputs_unlabeled_soft.detach(), dim=1, keepdim=False)
            model1.train()

            # get pseudo labels from model2 for unlabeled data
            model2.eval()
            with torch.no_grad():
                outputs_unlabeled  = model2(input_ul)
                outputs_unlabeled_soft = torch.softmax(outputs_unlabeled, dim=1)
                pseudo_labels_s2 = torch.argmax(outputs_unlabeled_soft.detach(), dim=1, keepdim=False)
            model2.train()

            # exchange pseudo label
            pseudo_labels_1 = pseudo_labels_s2
            pseudo_labels_2 = pseudo_labels_s1


            # generate syn for model 1
            syn_mask = lung_ul
            syn_mask[pseudo_labels_1>0] = 3
            syn_output_list =[]
            len_syn = input_ul.shape[0]
            for syn_idx in range(len_syn):
                with torch.no_grad():
                    normalize_input = transforms.functional.normalize(input_ul[syn_idx], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    output_ul_syn = syn_model(normalize_input.unsqueeze_(0), syn_mask[syn_idx:syn_idx+1].unsqueeze_(1), mode='inference', data_path=[file_name_ul[syn_idx]], 
                            style_code=style_output, alpha=0)
                    output_ul_syn_numpy = output_ul_syn[0].detach()
                    output_ul_syn_numpy = (output_ul_syn_numpy + 1) / 2.0
                    syn_output_list.append((output_ul_syn_numpy).unsqueeze_(0))
            syn_output_1 = torch.cat(syn_output_list, 0)

            # generate syn for model 2
            syn_mask = lung_ul
            syn_mask[pseudo_labels_2>0] = 3
            syn_output_list =[]
            len_syn = input_ul.shape[0]
            for syn_idx in range(len_syn):
                with torch.no_grad():
                    normalize_input = transforms.functional.normalize(input_ul[syn_idx], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    output_ul_syn = syn_model(normalize_input.unsqueeze_(0), syn_mask[syn_idx:syn_idx+1].unsqueeze_(1), mode='inference', data_path=[file_name_ul[syn_idx]], 
                            style_code=style_output, alpha=alpha)
                    output_ul_syn_numpy = output_ul_syn[0].detach()
                    output_ul_syn_numpy = (output_ul_syn_numpy + 1) / 2.0
                    syn_output_list.append((output_ul_syn_numpy).unsqueeze_(0))
            syn_output_2 = torch.cat(syn_output_list, 0)



            # train model 1
            volume_batch = torch.cat([input_l, syn_output_1, input_ul], 0)
            label_batch = torch.cat([target_l, pseudo_labels_1, pseudo_labels_1], 0)

            outputs_1 = model1(volume_batch)
            outputs_soft_1 = torch.softmax(outputs_1, dim=1)

            labeled_loss_1 = 0.5 * (ce_loss(outputs_1[:args.batch_size_label], label_batch[:][:args.batch_size_label].long()) + dice_loss(
                outputs_soft_1[:args.batch_size_label], label_batch[:args.batch_size_label].unsqueeze(1)))
            syn_supervision_1 =  0.5 * (ce_loss(outputs_1[args.batch_size_label:args.batch_size_label*2], label_batch[args.batch_size_label:args.batch_size_label*2].long()) + dice_loss(
                outputs_soft_1[args.batch_size_label:args.batch_size_label*2], label_batch[args.batch_size_label:args.batch_size_label*2].unsqueeze(1)))
            pseudo_supervision_1 =  0.5 * (ce_loss(outputs_1[args.batch_size_label*2:], label_batch[args.batch_size_label*2:].long()) + dice_loss(
                outputs_soft_1[args.batch_size_label*2:], label_batch[args.batch_size_label*2:].unsqueeze(1)))
        
            # train model 2
            volume_batch = torch.cat([input_l, syn_output_2, input_ul], 0)
            label_batch = torch.cat([target_l, pseudo_labels_2, pseudo_labels_2], 0)

            outputs_2 = model2(volume_batch)
            outputs_soft_2 = torch.softmax(outputs_2, dim=1)

            labeled_loss_2 = 0.5 * (ce_loss(outputs_2[:args.batch_size_label], label_batch[:][:args.batch_size_label].long()) + dice_loss(
                outputs_soft_2[:args.batch_size_label], label_batch[:args.batch_size_label].unsqueeze(1)))
            syn_supervision_2 =  0.5 * (ce_loss(outputs_2[args.batch_size_label:args.batch_size_label*2], label_batch[:][args.batch_size_label:args.batch_size_label*2].long()) + dice_loss(
                outputs_soft_2[args.batch_size_label:args.batch_size_label*2], label_batch[args.batch_size_label:args.batch_size_label*2].unsqueeze(1)))
            pseudo_supervision_2 =  0.5 * (ce_loss(outputs_2[args.batch_size_label*2:], label_batch[:][args.batch_size_label*2:].long()) + dice_loss(
                outputs_soft_2[args.batch_size_label*2:], label_batch[args.batch_size_label*2:].unsqueeze(1)))



            # calculate loss
            c_syn = args.consistency_syn
            c_pseudo = args.consistency_pseudo

            model1_loss = labeled_loss_1 + c_syn*syn_supervision_1 + c_pseudo*pseudo_supervision_1
            model2_loss = labeled_loss_2 + c_syn*syn_supervision_2 + c_pseudo*pseudo_supervision_2
            loss = model1_loss + model2_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()


            # write summary
            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))

    writer.close()


if __name__ == "__main__":
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