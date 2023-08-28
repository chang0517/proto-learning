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
import datetime
import math
from sklearn.metrics import confusion_matrix
import warnings
from utils import losses
import wandb 

class Proto_Trainer(object):
    def __init__(self, args, model=None,train_loader=None, val_loader=None,weighted_train_loader=None,per_class_num=[],log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.label_weighting = args.label_weighting
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.model = model
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), momentum=0.9, lr=self.lr,weight_decay=args.weight_decay)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.log = log
        self.beta = args.beta
        self.criterion_ce = None
        self.creterion_proto = None
        self.update_weight()
        self.get_loss()
        self.get_loss_proto()


    def get_loss(self):
        if self.args.loss == 'CE':
            self.criterion_ce = nn.CrossEntropyLoss(weight=self.per_cls_weights).cuda()
        elif self.args.loss == 'logit':
            self.criterion_ce = losses.LogitAdjust(self.cls_num_list)

    def get_loss_proto(self):
        if self.args.loss2 == 'proto':
            temp = 0.07
            self.creterion_proto = losses.SupConLoss_proto(self.cls_num_list, temperature=temp)  
        elif self.args.loss2 == 'supcon':
            temp = 0.07
            self.creterion_proto = losses.SupConLoss(temperature=temp)  
        elif self.args.loss2 == 'sim':
            self.creterion_proto = self.SimSiamLoss()

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            train_loader = iter(self.train_loader)

            for i, (inputs, targets) in enumerate(self.weighted_train_loader):

                bal_ce_input = inputs[0]
                bal_cont_input_1 = inputs[1]
                bal_cont_input_2 = inputs[2]
                bal_target = targets

                try:
                    input, target = next(train_loader)
                except:
                    weighted_train_loader = iter(self.train_loader)
                    input, target = next(weighted_train_loader)

                ce_input = input[0][:bal_ce_input.size()[0]]
                cont_input_1 = input[1][:bal_cont_input_1.size()[0]]
                cont_input_2 = input[2][:bal_cont_input_2.size()[0]]

                center_label = torch.Tensor([i for i in range(self.num_classes)]).cuda()

                ce_input = ce_input.cuda()
                cont_input_1 = cont_input_1.cuda()
                cont_input_2 = cont_input_2.cuda()
                target = target.cuda()
                
                bal_ce_input = bal_ce_input.cuda()
                bal_cont_input_1 = bal_cont_input_1.cuda()
                bal_cont_input_2 = bal_cont_input_2.cuda()
                bal_target = bal_target.cuda()

                if self.args.mixed == 'mixup_bal':
                    mixed_x, y_a, y_b, lam, index = mixup_data(bal_ce_input, bal_target)
                    _, logits, _  = self.model(mixed_x)
                    loss = mixup_criterion(self.criterion_ce, logits, y_a, y_b, lam)
                    balance_loss = loss
                
                elif self.args.mixed == 'cutmix_bal':
                    mixed_x, y_a, y_b, lam, index = cutmix_data(bal_ce_input, bal_target)
                    _, logits, _  = self.model(mixed_x)
                    loss = mixup_criterion(self.criterion_ce, logits, y_a, y_b, lam)
                    balance_loss = loss

                elif self.args.mixed == 'mixup':
                    mixed_x, lam = util.proto_mixup(ce_input,bal_ce_input)
                    _, logits, _  = self.model(mixed_x)
                    loss = util.mixup_criterion(self.criterion_ce, logits, target, bal_target, lam)
                    
                    balance_loss = loss
                
                elif self.args.mixed == 'cutmix':
                    cutmixed_x, lam = util.proto_cutmix(ce_input,bal_ce_input)
                    _, logits, _  = self.model(cutmixed_x)
                    loss = util.mixup_criterion(self.criterion_ce, logits, target, bal_target, lam)

                    balance_loss = loss
                
                elif self.args.mixed == 'both':
                    mix_x, lam = util.proto_mixup_cutmix(ce_input, bal_ce_input,alpha = self.args.alpha, beta = self.args.beta)
                    _, logits, _  = self.model(mix_x)
                    loss = util.mixup_criterion(self.criterion_ce, logits, target, bal_target, lam)
                    balance_loss = loss
                    
                elif self.args.mixed == 'ce':
                    _, logits, _  = self.model(bal_ce_input)
                    loss = self.criterion_ce(logits, bal_target)
                    balance_loss = loss
                # measure data loading time
                data_time.update(time.time() - end)

                # Data augmentation
                if self.args.use_proto:
                    images = torch.cat([cont_input_1, cont_input_2], dim=0)
                    feat_mlp, _, centers = self.model(images)
                    centers = centers[:len(self.cls_num_list)]
                    bsz = target.shape[0]
                    f1, f2 = torch.split(feat_mlp, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)           
                    if self.args.loss2 == 'supcon':
                        contrastive_loss = self.creterion_proto(features, target)   
                    else:
                        contrastive_loss = self.creterion_proto(center_label, centers, features, target)   
                        
                else:
                    _, logits_ce, _  = self.model(ce_input)
                    loss_ce = self.criterion_ce(logits_ce, target)
                    contrastive_loss = loss_ce
                loss = self.args.coeff_1 *   balance_loss +  self.args.coeff_2 * contrastive_loss
                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
                    # evaluate on validation set
            acc1, many_avg, med_avg, few_avg,  test_metrics = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)
            wandb.log({"Epoch": epoch})            
            wandb.log({"validation top1 avg": acc1})            
            wandb.log({"many avg": many_avg })
            wandb.log({"med avg":  med_avg})
            wandb.log({"few avg":  few_avg})
            wandb.log({"best avg":  best_acc1})   
            wandb.log(test_metrics)  

    def validate(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                _, output, _ = self.model(input)

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1 , flag='val', top1=top1, top5=top5))
            test_metrics = {f"class{idx}_acc":cls_acc[idx] for idx in range(len(cls_acc))}

            self.log.info(output)
            out_cls_acc = '%s Class Accuracy: %s' % (
            'val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))

            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            many_avg = float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps))
            med_avg = float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps))
            few_avg = float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
            
        return top1.avg, many_avg, med_avg, few_avg, test_metrics

    def SimSiamLoss(self,p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def paco_adjust_learning_rate(self,optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=50, beta=2):
    lam = np.random.beta(alpha, beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    y_a = y
    y_b = y[index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam, index

def cutmix_data_fix(x, lam, index):
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x

def mixup_data(x, y, alpha=1, beta=1, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def mixup_data_fix(x, lam, index, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

