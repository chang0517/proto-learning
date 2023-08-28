import argparse
 

# train set
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')

parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
parser.add_argument('--root', type=str, default='../data', help="dataset setting")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet_proto',choices=('resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'resnet32', 'resnet_proto'))
parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W',help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')
parser.add_argument('--dir_train_txt', type=str, default='../data/imagenet/ImageNet_LT_train.txt', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
parser.add_argument('--dir_test_txt', type=str, default='../data/imagenet/ImageNet_LT_test.txt', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")


parser.add_argument('--resample_weighting', default=0.2, type=float,help='weighted for sampling probability (q(1,k))')
parser.add_argument('--label_weighting', default=1.0, type=float, help='weighted for Loss')
parser.add_argument('--contrast_weight', default=10,type=int,help='Mixture Consistency  Weights')

parser.add_argument('--alpha', type=float, default=50, help="alpha")
parser.add_argument('--beta', type=float, default=2, help="beta")
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--use_proto', default=False, type=bool)
## ablation 
## 1. loss 1, loss 2 
## 2. cum learning
## 3. fix : 64, 128, 256, 512 
## 4. augmentation step 
## 5. channel number 

parser.add_argument('--method', default='GLMC', type=str, choices=('GLMC', 'proto'))
parser.add_argument('--loss', default="logit", type=str, help='loss type / method', choices=('CE',   'logit'))
parser.add_argument('--loss2', default="proto", type=str, help='loss type / method')
parser.add_argument('--aug_step', default="3", type=str, help='mixsup aug type')
parser.add_argument('--coeff_1', default= 1.0, type=float)
parser.add_argument('--coeff_2', default= 0.8, type=float)
parser.add_argument('--fix', default="Y", type=str)
parser.add_argument('--cum', default="Y", type=str)

parser.add_argument('--mix_coeff', default= 1, type=float)
parser.add_argument('--cut_coeff', default= 1, type=float)
parser.add_argument('--project', default= False, type=bool)
parser.add_argument('--mixed', default= 'both', type=str)
parser.add_argument('--weight_sample', default= False, type=bool)


parser.add_argument('--feat_dim', default=64 ,type=int, help='projection channel number')
parser.add_argument('--hidden_dim', default=512 ,type=int, help='projection channel number')
parser.add_argument('--nf', default=64 ,type=int, help='encoder')
parser.add_argument('--wandb', default='proto' ,type=str, help='encoder')

# etc.
parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N',help='print frequency (default: 100)')
parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--root_log', type=str, default='./output/')
parser.add_argument('--root_model', type=str, default='./output/')
parser.add_argument('--store_name', type=str, default='./output/')