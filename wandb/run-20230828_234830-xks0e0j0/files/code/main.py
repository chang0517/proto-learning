import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data
from imbalance_data import imbalance_cifar
import logging
import datetime
import math
from sklearn.metrics import confusion_matrix
from Proto_Trainer import Proto_Trainer
from torchvision.datasets import CIFAR10, CIFAR100
import wandb 
from arg import parser

best_acc1 = 0

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        print("=> creating model '{}'".format('resnext50_32x4d'))
        net = Resnet_LT.resnet_proto(cfg=args, num_class=args.num_classes, nf=args.nf, hidden_dim=args.hidden_dim, feat_dim=args.feat_dim)
        return net
    else:
        print("=> creating model '{}'".format(args.arch))
        net = ResNet_cifar.resnet_proto(args, num_class=args.num_classes, nf=args.nf, hidden_dim=args.hidden_dim, feat_dim=args.feat_dim)  
        return net

def get_dataset(args):
    transform_train,transform_val = util.get_transform(args, args.dataset, args.aug_step)
    if args.dataset == 'cifar10':
        if args.method == 'proto':
            trainset = imbalance_cifar.IMBALANCECIFAR10(root=os.path.join(args.root,'cifar-10-batches-py/'), download=True, transform=util.ThreeTransform(transform_train), imb_factor=args.imbanlance_rate, train=True)
        
        elif args.method == 'GLMC':
            trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(args.root,'cifar-100-python/'))
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.root,'cifar-10-batches-py/'),
                                download=True,
                                train=False,
                                transform=transform_val)
        print("load cifar10")
        return trainset,testset

    if args.dataset == 'cifar100':
        if args.method == 'proto':
            trainset = imbalance_cifar.IMBALANCECIFAR100(root=os.path.join(args.root,'cifar-100-python/'), download=True, transform=util.ThreeTransform(transform_train), imb_factor=args.imbanlance_rate, train=True)
            
        elif args.method == 'GLMC':
            trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(args.root,'cifar-100-python/'))
        testset = torchvision.datasets.CIFAR100(root=os.path.join(args.root,'cifar-100-python/'),
                                download=True,
                                train=False,
                                transform=transform_val)
        print("load cifar100")
        return trainset,testset

    elif args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt,util.TwoTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        return trainset,testset

    elif args.dataset == 'iNaturelist2018':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt,util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        return trainset,testset

def main():
    args = parser.parse_args()
    print(args)

    curr_time = datetime.datetime.now()
    args.store_name = '#'.join(["dataset: " + args.dataset, "arch: " + args.arch,"imbanlance_rate: " + str(args.imbanlance_rate)
            ,datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    store_name = get_store_name(args)
    # key = c79c81a7d90c296701ee71b2d8fdda688a58315b
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandb,
        name=store_name, 
    )              
    wandb.config.update(args)
    main_worker(args.gpu, args)

def main_worker(gpu, args):

    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    # Data loading code
    train_dataset,val_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    print(num_classes)
    assert num_classes == args.num_classes

    cls_num_list = train_dataset.get_cls_num_list()
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, #persistent_workers=True,
                                               pin_memory=True, 
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, 
                                             #persistent_workers=True,
                                             pin_memory=True)

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)
    train_sampler = None
    weighted_train_loader = None

    #weighted_loader (uniform distribution)
    cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.workers, 
                                                        #persistent_workers=True,
                                                        pin_memory=True,sampler=weighted_sampler)
    
    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    
    if args.method == 'proto':
        print("proto Training started!")
        if args.fix == 'Y':
            for p in model.module.fc.parameters():
                p.requires_grad = False
        trainer = Proto_Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list,log=logging)
        trainer.train()        
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))

if __name__ == '__main__':
    main()