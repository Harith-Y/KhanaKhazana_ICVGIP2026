import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch = labels.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            n_correct = correct[:k].reshape(-1).float().sum()
            results.append((n_correct * 100.0 / batch).item())
        return results
