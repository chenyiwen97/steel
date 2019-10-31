import torch
def dice_channel_torch(probability, truth, threshold):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for j in range(channel_num):
            channel_dice = dice_single_channel(probability[:, j,:,:], truth[:, j, :, :], threshold, batch_size)
            mean_dice_channel += channel_dice.sum(0)/(batch_size * channel_num)
    return mean_dice_channel

def dice_single_channel(probability, truth, threshold,  batch_size, eps = 1E-9):
    p = (probability.view(batch_size,-1) > threshold).float()
    t = (truth.view(batch_size,-1) > 0.5).float()
    dice = (2.0 * (p * t).sum(1) + eps)/ (p.sum(1) + t.sum(1) + eps)
    return dice