import torch
import numpy as np

# Helper
def adjust_learning_rate(epoch, optimizer):
    decay_epochs = np.array([20,40])
    decay_rate = 0.1
    steps = np.sum(epoch > decay_epochs)
    if steps > 0:
        new_lr = 0.05 * (decay_rate ** steps)
        #print(new_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def train(epoch, model, optimizer, train_data, criterion):
    adjust_learning_rate(epoch,optimizer)
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
                epoch, idx, len(train_data), loss=loss.item()))