import torch

def validate(epoch, model, val_data, criterion):
    # Logging variables
    sum = 0
    avg = 0
    print('eval ...')

    # Shift into evaluation mode
    model.eval()

    # Turn off gradient within this context
    with torch.no_grad():

        for idx, (input, target, _) in enumerate(val_data):

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

            # Logging
            sum += loss.item()
            avg = sum / (idx + 1)

            # print info
            if idx % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f}\t'.format(
                    epoch, idx, len(val_data), loss=avg))
    return avg