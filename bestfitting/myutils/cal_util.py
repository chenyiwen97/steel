from lossfuncs.loss import dice_score

# 用于存储和计算平均loss
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def metric(logit, truth, threshold=0.5):
    dice = dice_score(logit, truth, threshold=threshold)
    return dice