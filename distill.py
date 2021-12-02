import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Helper RepDistiller, https://github.com/HobbitLong/RepDistiller
class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

# Helper function to adjust learning rate, from paper
def adjust_learning_rate(epoch, optimizer, args):
    steps = np.sum(epoch > args.decay_e)
    if steps > 0:
        new_lr = args.lr * (args.decay_r ** steps)
        #print(new_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
            
def distill(epoch, args,  model_t, model_s, optimizer, train_data, criterion_s, criterion_div):
    losses = AverageMeter()
    adjust_learning_rate(epoch,optimizer, args)
    print('distillation ...')
    # Shift into train mode
    model_s.train()
    model_t.eval()
    for idx, (input, target, _) in enumerate(train_data):
        
        # Convert input data to a float
        input = input.float()

        # GPU Available?
        if torch.cuda.is_available():

            # Assign to default GPU
            input = input.to('cuda')
            target = target.to('cuda')

        # Forward    
        output_s = model_s(input)
        with torch.no_grad():
            output_t = model_t(input)

        # Loss
        loss_cls = criterion_s(output_s, target)
        loss_div = criterion_div(output_s, output_t)
        
        loss = 0.5 * loss_cls + 0.5 * loss_div
        losses.update(loss.item(), input.size(0))

        # Set the gradient of all optimized tensors to zero
        optimizer.zero_grad()

        # Back Prop
        loss.backward()

        # Perform a single optimization step
        optimizer.step()
        
        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.avg:.4f}\t'.format(
                epoch, idx, len(train_data), loss=losses))
    return losses.avg