import torch
import sys
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.mini_imagenet import ImageNet, MetaImageNet
from tqdm import tqdm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import scipy
from scipy.stats import t

# Helper Funcitons
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h
def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def main():
    # ImageNet class uses this notation for accessing 
    args = lambda x: None
    args.data_root = './data/miniImageNet'
    args.data_aug = True
    args.n_test_runs = 600 
    train_data = DataLoader(ImageNet(args=args,partition='train'), batch_size=64, shuffle=True, drop_last=True)
    val_data = DataLoader(ImageNet(args=args,partition='val'), batch_size=32)
    meta_test_data = DataLoader(MetaImageNet(args=args, partition='test', fix_seed=False), batch_size=1)
    meta_val_data = DataLoader(MetaImageNet(args=args, partition='val', fix_seed=False), batch_size=1)
    n_cls = 64

    model = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained = False, num_classes=n_cls)

    optimizer = optim.SGD(model.parameters(),lr=0.05,momentum=0.9,weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to('cuda')
        criterion = criterion.to('cuda')

        #optimize algo for given hardware
        cudnn.benchmark = True

    epoch = 1
    for i in range(1,epoch+1):
        # train
        print('train ...')
        model.train()
        ex = 0
        for input,target,_ in train_data:
            input = input.float()
            if torch.cuda.is_available():
                input = input.to('cuda')
                target = target.to('cuda')
            output = model(input)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print info
            if ex % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f}\t'.format(
                    epoch, ex, len(train_data), loss=loss.item()))
                sys.stdout.flush()
            ex += 1

        #validate
        print('eval ...')
        model.eval()
        # turn off gradient
        with torch.no_grad():
            for k, (input,target,_) in enumerate(val_data):
                input = input.float()
                if torch.cuda.is_available():
                    input = input.to('cuda')
                    target = target.to('cuda')
                output = model(input)
                loss = criterion(output,target)
                # print info
                if k % 100 == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss:.4f}\t'.format(
                        epoch, k, len(train_data), loss=loss.item()))
                    sys.stdout.flush()
        model.eval()
        acc = []
        with torch.no_grad():
            for idx, data in tqdm(enumerate(meta_test_data)):
                support_xs, support_ys, query_xs, query_ys = data
                support_xs = support_xs.to('cuda')
                query_xs = query_xs.to('cuda')
                batch_size, _, channel, height, width = support_xs.size()
                support_xs = support_xs.view(-1, channel, height, width)
                query_xs = query_xs.view(-1, channel, height, width)
                
                support_features = model(support_xs).view(support_xs.size(0), -1)
                query_features = model(query_xs).view(query_xs.size(0), -1)
                
                support_features = normalize(support_features)
                query_features = normalize(query_features)

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()

                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()
                clf = LogisticRegression(penalty='l2',
                                            random_state=0,
                                            C=1.0,
                                            solver='lbfgs',
                                            max_iter=1000,
                                            multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)

                acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
        acc, std = mean_confidence_interval(acc)
        print('test_acc: {:.4f}, test_std: {:.4f}'.format(acc, std))

if __name__ == '__main__':
    main()