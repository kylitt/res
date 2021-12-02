import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.mini_imagenet import ImageNet, MetaImageNet
from train import train
from validate import validate
from test import test
from distill import distill
from pytorch_resnet import resnet18
import numpy as np


def main():
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60, metavar='INT', help='number of epochs to run')
    parser.add_argument('--distill', type=int, default=0, metavar='INT', help='number of distillation runs')
    parser.add_argument('--val', type=bool, default=False, metavar='BOOL', help='run validation each epoch')

    parser.add_argument('--pretrain', type=bool, default=False, metavar='BOOL', help='load a resnet model from ./models')

    parser.add_argument('--lr', type=float, default=0.05, metavar='FLOAT', help='learning rate')
    parser.add_argument('--decay_e', type=str, default='20,40', metavar='LIST', help='what epochs to decay the learning rate')
    parser.add_argument('--decay_r', type=float,  default=0.1, metavar='FLOAT', help='decay rate for learning rate')

    parser.add_argument('--weight', type=float, default=5e-4, metavar='FLOAT', help='SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='FLOAT', help='SGD momentum')
    parser.add_argument('--batch_s', type=int, default=64, metavar='INT', help='SGD batch size')

    args = parser.parse_args()

    # From the paper, used for the data loading 
    args.data_aug = True
    args.data_root = './data/miniImageNet'
    args.n_test_runs = 600

    # parse input string into a numpy array
    args.decay_e = np.fromstring(args.decay_e, dtype=int, sep=',')

    # Load Data
    train_data = DataLoader(ImageNet(args=args, partition='train'), batch_size=args.batch_s, shuffle=True, drop_last=True)
    meta_test_data = DataLoader(MetaImageNet(args=args, partition='test', fix_seed=False), batch_size=1)
    if args.val:
        val_data = DataLoader(ImageNet(args=args, partition='val'), batch_size=args.batch_s // 2)

    # Number of classes
    n_cls = 64

    # Load the Pytorch ResNet18 model, adapted to a ResNet12
    model = resnet18(pretrained = args.pretrain, num_classes=n_cls)

    # Create a Stochastic Gradient Descent optimizer with the given parameters
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight)

    # Cross Entropy Loss 
    criterion = nn.CrossEntropyLoss()

    # GPU Available?
    if torch.cuda.is_available():

        # Multiple GPU?
        if torch.cuda.device_count() > 1:
            model_t = nn.DataParallel(model)

        # else assigns to the default GPU
        model = model.to('cuda')
        criterion = criterion.to('cuda')

        # Optimize algorithm for the given hardware
        cudnn.benchmark = True

    # train
    if not args.pretrain:
        for i in range(1,args.epoch+1):
            # train teacher
            train(i, args,  model, optimizer, train_data, criterion)

            #validate teacher
            if args.val:
                validate(i, model, val_data, criterion)

        # does not support multiple gpu
        state = {'model': model.state_dict()}
        torch.save(state,'./models/resnet_simple.pth')

    # distill
    # set model as teacher
    model_t = model
    for j in range(args.distill):
        # Sudent model
        model_s = resnet18(pretrained = False, num_classes=n_cls)

        # Create a Stochastic Gradient Descent optimizer with the given parameters
        optimizer_s = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight)

        # Cross Entropy Loss 
        criterion_s = nn.CrossEntropyLoss()
        
        # GPU Available?
        if torch.cuda.is_available():

            # Multiple GPU?
            if torch.cuda.device_count() > 1:
                model_t = nn.DataParallel(model_t)
                model_s = nn.DataParallel(model_s)

            # else assigns to the default GPU
            model_t = model_t.to('cuda')
            model_s = model_s.to('cuda')
            criterion_s = criterion_s.to('cuda')

            # Optimize algorithm for the given hardware
            cudnn.benchmark = True
        for j in range(1,args.epoch+1):
            # distill from teacher to student
            distill(j, args,  model_t, model_s, optimizer_s, train_data, criterion_s)

            #validate student
            if args.val:
                validate(j, model_s, val_data, criterion_s)

        # Teacher model
        model_t = model_s

        # does not support multiple gpu
        state = {'model': model_t.state_dict()}
        torch.save(state,'./models/resnet_dist.pth')

    # test final model
    test(model_t, meta_test_data)

if __name__ == '__main__':
    main()