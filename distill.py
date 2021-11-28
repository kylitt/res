import torch
import torch.nn.functional as F
import sys

# helper function
def div(output_s, output_t):
    temperature = 4
    s = F.log_softmax(output_s/temperature, dim=1)
    t = F.softmax(output_t/temperature, dim=1)
    loss = F.kl_div(s, t, reduction='sum') * (temperature**2) / s.shape[0]
    return loss

def distill(i, model_t, model_s, optimizer, train_data, criterion):
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
        loss_cls = criterion(output_s, target)
        loss_div = div(output_s, output_t)
        
        loss = 0.5 * loss_cls + 0.5 * loss_div


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
                i, idx, len(train_data), loss=loss.item()))