# Prototype based Supervised Contrastive Learning using Fixed Classifier for Long-Tailed Recognition
This is a Pytorch implementation of "Prototype based Supervised Contrastive Learning using Fixed Classifier for Long-Tailed Recognition"

## Update 2023/08/29
> 
The experimental setup was as follows:

````
python main.py --dataset cifar10 --num_classes 10 --arch resnet_proto --nf 64 --weight_sample True --method proto --fix True --loss logit --mixed both_bal --use_proto True --project True --weight_decay 5e-3 --lr 0.01
````

### CIFAR10-LT
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|
| PSCL   | 100   | ResNet-32     | 89.09%    |
| PSCL   | 50    | ResNet-32     | 91.66%    |
| PSCL   | 10    | ResNet-32     | 94.53%    |
