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


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/code/SSL/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Test', help='experiment_name')
parser.add_argument('--model', type=str, default='unet2', help='model_name')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_per', type=float, default=0.1, help='percent of labeled data')

if False:
    parser.add_argument('--dataset_name', type=str, default='COVID249', help='Name of dataset')
    parser.add_argument('--model_path', type=str, default='/home/code/SSL/exp/COVID249/model.pth', help='path of teacher model')
else:
    parser.add_argument('--dataset_name', type=str, default='MOS1000', help='Name of dataset')
    parser.add_argument('--model_path', type=str, default='/home/code/SSL/exp/MOS1000/model.pth', help='path of teacher model')
args = parser.parse_args()



def test(args, snapshot_path):
    model = net_factory(net_type=args.model)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    nsd, dice = get_model_metric(args = args, model = model, snapshot_path=snapshot_path, model_name='model', mode='test')
    print('nsd : %f dice : %f ' % (nsd, dice))



if __name__ == "__main__":
    snapshot_path = "{}exp/{}/test_{}_{}_{}".format(args.root_path, args.dataset_name, args.exp, args.labeled_per, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    test(args, snapshot_path)