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
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs to run')
    parser.add_argument('--distill', type=int, default=0, help='number of distillation runs')

    parser.add_argument('--lr', type=float, default=0.05,help='learning rate')
    parser.add_argument('--decay_e', type=str, default='20,40', metavar='A,B,C...', help='what epochs to decay the learning rate')
    parser.add_argument('--decay_r', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--weight', type=float, default=5e-4, help='SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--batch_s', type=int, default=64, help='SGD batch size')

    # From the paper, used for the data loading 
    parser.add_argument('--data_root', type=str, default='./data/miniImageNet', metavar='D', help='path to data root')
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',help='number of test runs')

    args = parser.parse_args()
    args.data_aug = True

    iterations = args.decay_e.split(',')
    args.decay_e = list([])
    for it in iterations:
        args.decay_e.append(int(it))
    args.decay_e = np.array(args.decay_e)
    # ImageNet/MetaImageNet class uses this notation for accessing attributes
    # args = lambda x: None
    # args.data_root = './data/miniImageNet'
    # args.data_aug = True
    # args.n_test_runs = 600

    # Load Data
    train_data = DataLoader(ImageNet(args=args, partition='train'), batch_size=args.batch_s, shuffle=True, drop_last=True)
    val_data = DataLoader(ImageNet(args=args, partition='val'), batch_size=args.batch_s // 2)
    meta_test_data = DataLoader(MetaImageNet(args=args, partition='test', fix_seed=False), batch_size=1)

    # Number of classes
    n_cls = 64

    # Load the Pytorch ResNet18 model, adapted to a ResNet12
    model = resnet18(pretrained = False, num_classes=n_cls)

    # Create a Stochastic Gradient Descent optimizer with the given parameters
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight)

    # Cross Entropy Loss 
    criterion = nn.CrossEntropyLoss()

    # GPU Available?
    if torch.cuda.is_available():

        # Multiple GPU?
        if torch.cuda.device_count() > 1:
            model_t = nn.DataParallel(model)
            #model_s = nn.DataParallel(model_s)

        # else assigns to the default GPU
        model = model.to('cuda')
        criterion = criterion.to('cuda')

        # Optimize algorithm for the given hardware
        cudnn.benchmark = True

    # train
    for i in range(1,args.epoch+1):
        # train teacher
        train(i, args,  model, optimizer, train_data, criterion)

        #validate teacher
        #validate(i, model, val_data, criterion)

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
            #validate(j, model_s, val_data, criterion_s)

        # Teacher model
        model_t = model_s

    # does not support multiple gpu
    state = {'model': model_t.state_dict()}
    torch.save(state,'./models/resnet_dist.pth')

    # test final model
    test(model_t, meta_test_data)

if __name__ == '__main__':
    main()