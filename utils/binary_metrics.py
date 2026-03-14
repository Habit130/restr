import torch


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


class BinarySegMeter:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.sample_count = 0

    def update(self, prediction, target):
        prediction = prediction.bool()
        target = target.bool()

        self.tp += int(torch.logical_and(prediction, target).sum().item())
        self.fp += int(torch.logical_and(prediction, torch.logical_not(target)).sum().item())
        self.fn += int(torch.logical_and(torch.logical_not(prediction), target).sum().item())
        self.tn += int(torch.logical_and(torch.logical_not(prediction), torch.logical_not(target)).sum().item())
        self.sample_count += int(prediction.shape[0])

    def compute(self):
        fg_iou = safe_div(self.tp, self.tp + self.fp + self.fn)
        bg_iou = safe_div(self.tn, self.tn + self.fp + self.fn)
        dice = safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        recall = safe_div(self.tp, self.tp + self.fn)
        fg_acc = safe_div(self.tp, self.tp + self.fn)
        bg_acc = safe_div(self.tn, self.tn + self.fp)

        return {
            "IoU": fg_iou,
            "Dice": dice,
            "Recall": recall,
            "mIoU": (fg_iou + bg_iou) / 2.0,
            "mAcc": (fg_acc + bg_acc) / 2.0,
            "sample_count": self.sample_count,
        }
