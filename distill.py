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
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
def distill(epoch, args,  model_t, model_s, optimizer, train_data, criterion_s, criterion_div):
    # Logging variables
    sum = 0
    avg = 0

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

        # Logging
        sum += loss.item()
        avg = sum / (idx + 1)

        # Set the gradient of all optimized tensors to zero
        optimizer.zero_grad()

        # Back Prop
        loss.backward()

        # Perform a single optimization step
        optimizer.step()
        
        # print info every 100 inputs
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss:.4f}\t'.format(
                epoch, idx, len(train_data), loss = avg))
    return avg