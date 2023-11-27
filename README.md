# Prototype based Supervised Contrastive Learning using Fixed Classifier for Long-Tailed Recognition
This is a Pytorch implementation of "Prototype based Supervised Contrastive Learning using Fixed Classifier for Long-Tailed Recognition"

The experimental setup for CIFAR10-LT Dataset is as follows:

````
python main.py --dataset cifar10 --num_classes 10 --weight_sample True --method proto --fix True --loss logit --mixed both_bal --use_proto True --project True --weight_decay 5e-3 --lr 0.01
````

The experimental setup for CIFAR100-LT Dataset is as follows:

````
python main.py --dataset cifar100 --num_classes 100 --weight_sample True --method proto --fix True --loss logit --mixed both_bal --use_proto True --project True --weight_decay 5e-3 --lr 0.01
````

The experimental setup for ImageNet-LT Dataset is as follows:

````
python main.py --dataset ImageNet-LT --num_classes 1000 --weight_sample True --method proto --fix True --loss logit --mixed both_bal --use_proto True --project True --weight_decay 5e-3 --lr 0.01
````
### CIFAR10-LT
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|
| PSCL   | 100   | ResNet-32     | 89.22%    |
| PSCL   | 50    | ResNet-32     | 92.34%    |
| PSCL   | 10    | ResNet-32     | 95.52%    |

### CIFAR100-LT
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|
| PSCL   | 100   | ResNet-32     | 59.61%    |
| PSCL   | 50    | ResNet-32     | 64.14%    |
| PSCL   | 10    | ResNet-32     | 74.34%    |

### ImageNet-LT
| Method | Model | Top-1 Acc(%) |
| :---:|:---:|:---:|
| PSCL   | ResNet-50     | 58.47%    |
| PSCL   | ResNeXt-50     | 58.58%    |

