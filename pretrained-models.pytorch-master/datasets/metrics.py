from __future__ import division
import torch
from abc import ABCMeta, abstractproperty

def _to_value(other):
    if isinstance(other, Metric):
        other = other.value
    return other

class Metric(metaclass=ABCMeta):

    @abstractproperty
    def value(self):
        raise NotImplementedError

    def __gt__(self, other):
        return self.value > _to_value(other)

    def __lt__(self, other):
        return self.value < _to_value(other)

    def __le__(self, other):
        return self.value <= _to_value(other)

    def __ge__(self, other):
        return self.value >= _to_value(other)

class Accuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct = 0
        self.count = 0

    def update(self, output: torch.Tensor, target: torch.Tensor):
        assert output.size(0) == target.size(0)

        with torch.no_grad():
            _, pred = output.topk(self.top_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[:self.top_k].view(-1).float().sum(0, keepdim=True).item()

        self.correct += correct_k
        self.count += output.size(0)

    @property
    def value(self):
        return 100 * self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.value)

class Average(Metric):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number

    @property
    def value(self):
        if self.count == 0:
            return float('inf')
        else:
            return self.sum / self.count

    def __str__(self):
        return '{:.4f}'.format(self.value)
