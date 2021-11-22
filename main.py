import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.mini_imagenet import ImageNet, MetaImageNet
from train import train
from validate import validate
from test import test


def main():
    # ImageNet/MetaImageNet class uses this notation for accessing attributes
    args = lambda x: None
    args.data_root = './data/miniImageNet'
    args.data_aug = True
    args.n_test_runs = 600

    # Load Data
    train_data = DataLoader(ImageNet(args=args,partition='train'), batch_size=64, shuffle=True, drop_last=True)
    val_data = DataLoader(ImageNet(args=args,partition='val'), batch_size=32)
    meta_test_data = DataLoader(MetaImageNet(args=args, partition='test', fix_seed=False), batch_size=1)

    # Number of classes
    n_cls = 64

    # Load the Pytorch ResNet18 model
    model = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained = False, num_classes=n_cls)

    # Create a Stochastic Gradient Descent optimizer with the given parameters
    optimizer = optim.SGD(model.parameters(),lr=0.05,momentum=0.9,weight_decay=5e-4)

    # Cross Entropy Loss 
    criterion = nn.CrossEntropyLoss()

    # GPU Available?
    if torch.cuda.is_available():

        # Multiple GPU?
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # else assigns to the default GPU
        model = model.to('cuda')
        criterion = criterion.to('cuda')

        # Optimize algorithm for the given hardware
        cudnn.benchmark = True

    epoch = 10
    for i in range(1,epoch+1):
        # train
        train(i, model,optimizer,train_data,criterion)

        #validate
        validate(i, model, val_data, criterion)

    #test
    test(model,meta_test_data)

if __name__ == '__main__':
    main()