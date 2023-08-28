import os
import argparse
import torch
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data
import torch.nn.functional as F
from arg import parser

def validate(model,val_loader,args):
    top1 = AverageMeter('Acc@1', ':6.2f')
    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            _, output, _ = model(input)
            # measure accuracy
            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target)) 
        
            output = 'Testing:  ' + str(i) + ' Prec@1:  ' + str(top1.val)
            print(output, end="\r")
        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        acc1 = mic_acc_cal(preds[total_labels != -1],
                                    total_labels[total_labels != -1])
    
    return acc1
        # output = ('{flag} Results: Prec@1 {top1.avg:.3f}'.format(flag='val', top1=top1))
        # print(output)

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        net = Resnet_LT.resnet_proto(cfg=args, num_class=args.num_classes, 
                                    nf=args.nf, 
                                    hidden_dim=args.hidden_dim, feat_dim=args.feat_dim)  
     
        print("=> creating model '{}'".format('resnext50_32x4d'))
    else:
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
        elif args.arch == 'resnet_proto':
            net = ResNet_cifar.resnet_proto(args,num_class=args.num_classes, 
                                            nf=args.nf, 
                                            hidden_dim=args.hidden_dim, feat_dim=args.feat_dim)  

        print("=> creating model '{}'".format(args.arch))
    return net

def get_dataset(args):
    _,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-10-batches-py/'))
        print("load cifar10")
        return testset

    if args.dataset == 'cifar100':
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
        print("load cifar100")
        return testset

    if args.dataset == 'ImageNet-LT':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        print("load ImageNet-LT")
        return testset

    if args.dataset == 'iNaturelist2018':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        print("load iNaturelist2018")
        return testset

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
    # create model
    model = get_model(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # test from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    
    print("Testing started!")
    # switch to evaluate mode
    model.eval()
    
    record_list=[]
    if args.dataset.startswith('cifar'):
        distrb = {
                'uniform': (0, False),
                'forward50': (0.02, False),
                'forward25': (0.04, False), 
                'forward10':(0.1, False),
                'forward5': (0.2, False),
                'forward2': (0.5, False),
                'backward50': (0.02, True),
                'backward25': (0.04, True),
                'backward10': (0.1, True),
                'backward5': (0.2, True),
                'backward2': (0.5, True),
            }  
    
        record_list=[]
        test_distribution_set = ["forward50",  "forward25", "forward10", "forward5", "forward2", "uniform", "backward2", "backward5", "backward10", "backward25", "backward50"] 
        for test_distribution in test_distribution_set: 
            # print(test_distribution)
            _, transform_val = util.get_transform(args, args.dataset, args.aug_step)
            if args.dataset == 'cifar10':
                testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=distrb[test_distribution][0], train=False, reverse=distrb[test_distribution][1], transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/', reverse=test_distribution[1]))
            elif args.dataset == 'cifar100':
                testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=distrb[test_distribution][0], train=False, reverse=distrb[test_distribution][1], transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.workers, 
                                            #persistent_workers=True,
                                            pin_memory=True)
            acc1, many_avg, med_avg, few_avg = validate(val_loader=val_loader)
            record = [acc1*100, many_avg, med_avg, few_avg]
        
            record_list.append(record)
        print('='*25, ' Final results ', '='*25)
        i = 0
        for txt in record_list:
            print(test_distribution_set[i]+'\t')
            print(*txt)          
            i+=1
    elif args.dataset == 'ImageNet-LT':
        test_distribution_set = ["forward50",  "forward25", "forward10", "forward5", "forward2", "uniform",  "backward2", "backward5", "backward10", "backward25", "backward50"] 
        record_list = []
        for test_distribution in test_distribution_set:
            test_txt  = './data_txt/ImageNet_LT_%s.txt'%(test_distribution) 
            print(test_txt)
            _, transform_val = util.get_transform(args, args.dataset, args.aug_step)
            
            testset = dataset_lt_data.LT_Dataset(args.root, test_txt, transform_val)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)
            record = validate(model, val_loader, args)
            record_list.append(record)
        print('='*25, ' Final results ', '='*25)
        i = 0
        for txt in record_list:
            print(test_distribution_set[i]+'\t')
            print(txt)          
            i+=1
if __name__ == '__main__':
    # test set
    main()