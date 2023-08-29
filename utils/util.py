import shutil
from torch.utils import data
import copy
import os
from imbalance_data.cifar100Imbanlance import *
from imbalance_data.cifar10Imbanlance import *
from imbalance_data.dataset_lt_data import *
import utils.moco_loader as moco_loader
from utils.randaugment import rand_augment_transform
from utils.autoaug import CIFAR10Policy, Cutout

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform[0](x), self.transform[1](x)]

class ThreeTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform[0](x), self.transform[1](x), self.transform[2](x)]

class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_transform(args, dataset, aug_step=3):
    if dataset == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
        normalize = transforms.Normalize(mean,std)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    
        augmentation_regular_P = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=16),
            normalize,
        ])

        augmentation_regular_ALL = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            normalize,
        ])        

        augmentation_sim_cifar = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        
        if aug_step == "0":
            return transform_train, transform_val
        
        if aug_step == "1":
            transform_train = [augmentation_sim_cifar, augmentation_sim_cifar, augmentation_sim_cifar]      
        elif aug_step == "2":
            transform_train = [augmentation_regular_P, augmentation_sim_cifar, augmentation_sim_cifar]
        elif aug_step == "3":
            transform_train = [augmentation_regular_ALL, augmentation_sim_cifar, augmentation_sim_cifar]
        
        return transform_train,transform_val

    if dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
        augmentation_regular_P = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=16),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        augmentation_regular_ALL = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])        

        augmentation_sim_cifar = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        if aug_step == "0":
            return transform_train, transform_val
        
        if aug_step == "1":
            transform_train = [augmentation_sim_cifar, augmentation_sim_cifar, augmentation_sim_cifar]      
        elif aug_step == "2":
            transform_train = [augmentation_regular_P, augmentation_sim_cifar, augmentation_sim_cifar]
        elif aug_step == "3":
            transform_train = [augmentation_regular_ALL, augmentation_sim_cifar, augmentation_sim_cifar]
        return transform_train, transform_val
    
    elif dataset == "ImageNet-LT":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        augmentation_sim = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        augmentation_randncls = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ])
        augmentation_randnclsstack = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
            ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        
        if args.arch == 'resnet50':
            transform_train = [augmentation_randncls, augmentation_sim]
        elif args.arch == 'resnext50_32x4d':
            transform_train = [augmentation_randnclsstack, augmentation_sim]
            
        return transform_train, transform_val

        return transform_train, transform_val

    if dataset == "iNaturelist2018":
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best, epoch):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if epoch % 20 == 0:
        filename = '%s/%s/%s_ckpt.pth.tar' % (args.root_model, args.store_name, str(epoch))
        torch.save(state, filename)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def GLMC_mixed(org1, org2, invs1, invs2, label_org, label_invs, label_org_w, label_invs_w, alpha=1):
    lam = np.random.beta(alpha, alpha)

    # mixup
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    # cutmix
    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)
    org2[:, :, bbx1:bbx2, bby1:bby2] = invs2[:, :, bbx1:bbx2, bby1:bby2]

    lam_cutmix = lam
    cutmix_y = lam_cutmix * label_org + (1 - lam_cutmix) * label_invs
    cutmix_y_w = lam_cutmix * label_org_w + (1 - lam_cutmix) * label_invs_w

    return mixup_x, org2, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w

def print_model_param_nums(model=None):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total


def proto_mixup_cutmix(input,bal_input, alpha=20, beta=2):
    lam = np.random.beta(alpha, beta)
    # lam = 0.95 

    mixup_x = lam * input + (1 - lam) * bal_input
    bbx1, bby1, bbx2, bby2 = rand_bbox(bal_input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = bal_input[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, lam

def proto_both_bal(x, y, alpha=20, beta=2):
    lam = np.random.beta(alpha, beta)
    # lam = 0.95 
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()
    
    mixup_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    cut_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return mixup_x, x, y_a, y_b, lam, cut_lam, index

def proto_mixup(input, bal_input, alpha=1, beta=1):
    lam = np.random.beta(alpha, beta)
    # lam = 0.95 

    mixup_x = lam * input + (1 - lam) * bal_input
    return mixup_x, lam

def proto_cutmix(input, bal_input, alpha=1, beta=1):
    lam = np.random.beta(alpha, beta)
    # lam = 0.95 
    bbx1, bby1, bbx2, bby2 = rand_bbox(bal_input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = bal_input[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_store_name(args):
    base_name = '+'.join([args.dataset, str(args.imbanlance_rate), "nf_"+str(args.nf)])
    if args.coeff_2 == 0 & (args.weight_sample == False) & (args.mixed == 'ce'):
        store_name = '+'.join([base_name,'LT_CE'])
    elif (args.weight_sample == True) & (args.mixed == 'ce'):
        store_name = '+'.join([base_name,'Bal_CE'])
    elif (args.weight_sample == True) & (args.mixed == 'ce') & (args.use_proto == False):
        store_name = '+'.join([base_name,'LT_Bal_CE'])
    elif (args.mixed == 'both_bal') & (args.use_proto == True):
        store_name = '+'.join([base_name,'LT_Bal_both_bal'])
    
    elif args.coeff_2 == 0 & (args.weight_sample ==False) & (args.mixed == 'mixup_bal'):
        store_name = '+'.join([base_name,'LT_mixup'])
    elif (args.weight_sample ==True) & (args.mixed == 'mixup_bal'):
        store_name = '+'.join([base_name,'Bal_mixup'])
    elif (args.weight_sample == True) & (args.mixed == 'both'):
        store_name = '+'.join([base_name,'both'])
    
    if args.use_proto:
        store_name = '+'.join([store_name, 'proto'])
    
    return store_name
        
