import torch
import numpy as np

# Helper function to adjust learning rate, from paper
def adjust_learning_rate(epoch, optimizer, args):
    steps = np.sum(epoch > args.decay_e)
    if steps > 0:
        new_lr = args.lr * (args.decay_r ** steps)
        #print(new_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def train(epoch, args, model, optimizer, train_data, criterion):
    # Logging variables
    sum = 0
    avg = 0
    adjust_learning_rate(epoch,optimizer, args)
    print('train ...')

    # Shift into train mode
    model.train()

    for idx, (input, target, _) in enumerate(train_data):
        # Convert input data to a float
        input = input.float()

        # GPU Available?
        if torch.cuda.is_available():

            # Assign to default GPU
            input = input.to('cuda')
            target = target.to('cuda')

        # Forward    
        output = model(input)

        # Loss
        loss = criterion(output, target)
        sum += loss.item()
        avg = sum / (idx + 1) 

        # Set the gradient of all optimized tensors to zero
        optimizer.zero_grad()

        # Back Prop
        loss.backward()

        # Perform a single optimization step
        optimizer.step()
        
        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss:.4f}\t'.format(
                epoch, idx, len(train_data), loss=avg))

    return avg