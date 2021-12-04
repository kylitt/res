import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import re

# Our local code
from train import train
from validate import validate
from test import test
from distill import distill, DistillKL
from pytorch_resnet import resnet18

# Local code from the paper, https://github.com/WangYueFt/rfs
from dataset.mini_imagenet import ImageNet, MetaImageNet

def main():
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, metavar='INT', help='number of epochs to run')
    parser.add_argument('--distill', type=int, default=0, metavar='INT', help='number of distillation runs')
    parser.add_argument('--val', type=bool, default=False, metavar='BOOL', help='run validation each epoch')
    parser.add_argument('--save_freq', type=int, default=50, metavar='INT', help='how often to save')

    parser.add_argument('--model_path', type=str, default='', metavar='STR', help='path to a pretrained model')

    parser.add_argument('--n_shots', type=int, default=1, metavar='INT', choices=[1, 5])

    parser.add_argument('--lr', type=float, default=0.05, metavar='FLOAT', help='learning rate')
    parser.add_argument('--decay_e', type=str, default='60,80', metavar='LIST', help='what epochs to decay the learning rate')
    parser.add_argument('--decay_r', type=float,  default=0.1, metavar='FLOAT', help='decay rate for learning rate')

    parser.add_argument('--weight', type=float, default=5e-4, metavar='FLOAT', help='SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='FLOAT', help='SGD momentum')
    parser.add_argument('--batch_s', type=int, default=64, metavar='INT', help='SGD batch size')

    args = parser.parse_args()

    # From the paper, used for the data loading 
    args.data_aug = True
    args.data_root = './data/miniImageNet'
    args.n_test_runs = 600

    # Load Data
    train_data = DataLoader(ImageNet(args=args, partition='train'), batch_size=args.batch_s, shuffle=True, drop_last=True)
    meta_test_data = DataLoader(MetaImageNet(args=args, partition='test', fix_seed=False), batch_size=1)
    if args.val or args.model_path:
        val_data = DataLoader(ImageNet(args=args, partition='val'), batch_size=args.batch_s // 2)

    run_vars = ('epochs_{}_lr_{}_decay_e_{}_decay_r_{}_weight_{}_momentum_{}_batch_s_{}').format(args.epoch,args.lr,args.decay_e,args.decay_r,args.weight,args.momentum,args.batch_s)
    if args.model_path:
        # remove file type
        path = re.sub('.pth','',args.model_path)
    else:
        path = './models/resnet_' + run_vars
    # parse input string into a numpy array
    args.decay_e = np.fromstring(args.decay_e, dtype=int, sep=',')

    # Number of classes set for miniImageNet
    n_cls = 64

    # Load the Pytorch ResNet18 model, adapted to a ResNet12
    model = resnet18(model_path=args.model_path, num_classes=n_cls)

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
    if not args.model_path:
        # Create the Tensorboard Logger
        writer = SummaryWriter(log_dir=('runs/{}_pretraining').format(run_vars))
        for i in range(1,args.epoch+1):
            # train teacher
            loss = train(i, args,  model, optimizer, train_data, criterion)
            writer.add_scalar('Loss/train', loss, i)
            #validate teacher
            if args.val:
                eval = validate(i, model, val_data, criterion)
                writer.add_scalar('Loss/validate', eval, i)
            if(i % args.save_freq == 0):
                state = {'model': model.state_dict()}
                temp_path = ('_save_epoch_{}.pth').format(i)
                torch.save(state,path+temp_path)
        writer.flush()
        writer.close()
        # does not support multiple gpu
        state = {'model': model.state_dict()}
        torch.save(state,path+'.pth')
    else:
        print('Validate Loaded model...')
        validate(0, model, val_data, criterion)

    # distill
    # set model as teacher
    model_t = model
    for j in range(args.distill):
        # Create the Tensorboard Logger
        writer = SummaryWriter(log_dir=('runs/{}_distillation_{}').format(run_vars,j))

        # Sudent model
        model_s = resnet18(num_classes=n_cls)

        # Create a Stochastic Gradient Descent optimizer with the given parameters
        optimizer_s = optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight)

        # Cross Entropy Loss 
        criterion_s = nn.CrossEntropyLoss()

        # Divergence
        criterion_div = DistillKL(4)
        
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
            criterion_div = criterion_div.to('cuda')

            # Optimize algorithm for the given hardware
            cudnn.benchmark = True

        for k in range(1,args.epoch+1):
            # distill from teacher to student
            loss = distill(k, args,  model_t, model_s, optimizer_s, train_data, criterion_s, criterion_div)
            writer.add_scalar(('Loss/train').format(j), loss, k)

            #validate student
            if args.val:
                eval = validate(k, model_s, val_data, criterion_s)
                writer.add_scalar(('Loss/validate').format(j), eval, k)

            if(k % args.save_freq == 0):
                temp_path = path + ('_dist_ver_{}_save_epoch_{}.pth').format(j,k)
                state = {'model': model.state_dict()}
                torch.save(state,temp_path)

            writer.flush()
            writer.close()

        # Teacher model
        model_t = model_s

        # does not support multiple gpu
        temp_path = path + ('_dist_ver_{}.pth').format(j)
        state = {'model': model_t.state_dict()}
        torch.save(state,temp_path)

    # test final model
    test(model_t, meta_test_data)


if __name__ == '__main__':
    main()
