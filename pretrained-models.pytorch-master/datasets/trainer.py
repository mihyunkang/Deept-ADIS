
import os
import shutil
from abc import ABCMeta, abstractmethod

import mlconfig
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from .metrics import Accuracy, Average
from .deepfake_valid import DEEPFAKE_val_DataLoader
from .deepfake_train import DEEPFAKE_train_DataLoader


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, device: torch.device,
                 num_epochs: int, output_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        #데이터 로드!
        self.train_loader_1 = DEEPFAKE_train_DataLoader(0, batch_size=30)
        self.train_loader_2 = DEEPFAKE_train_DataLoader(1, batch_size=30)
        self.train_loader_3 = DEEPFAKE_train_DataLoader(2, batch_size=30)
        self.train_loader_4 = DEEPFAKE_train_DataLoader(3, batch_size=30)
        self.train_loader_5 = DEEPFAKE_train_DataLoader(4, batch_size=30) 
        # self.valid_loader = DEEPFAKE_val_DataLoader(batch_size=30)
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        epochs = trange(self.epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        for self.epoch in epochs:
            self.scheduler.step()
            train_loss_sum, train_acc_sum = 0.0, 0.0
            print("\n**********{}/5 fold**********".format(self.epoch%5))
            for fold in range(5):
                if self.epoch%5 != fold:
                    train_loss, train_acc = self.train(fold+1)
                    train_loss_sum += train_loss.value
                    train_acc_sum += train_acc.value
                else:
                    valid_loss, valid_acc = self.evaluate(fold+1)

            #트레이닝할때마다 저장하는 체크포인트
            self.save_checkpoint(os.path.join(self.output_dir, 'checkpoint.pth'))
            #best acc 일때 갱신하는 체크포인트 - 이걸로 test 할 것!
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc.value
                self.save_checkpoint(os.path.join(self.output_dir, 'best.pth'))

            epochs.set_postfix_str(f'train loss: {train_loss_sum/4}, train acc: {train_acc_sum/4}, '
                                   f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                   f'best valid acc: {self.best_acc:.2f}')

    def train(self, cross_val_num):
        self.model.train()
        #print(self.model)
        
        train_loss = Average()
        train_acc = Accuracy()
        if cross_val_num == 1:
            train_loader = tqdm(self.train_loader_1, ncols=0, desc='Train')
        elif cross_val_num == 2:
            train_loader = tqdm(self.train_loader_2, ncols=0, desc='Train')
        elif cross_val_num == 3:
            train_loader = tqdm(self.train_loader_3, ncols=0, desc='Train')
        elif cross_val_num == 4:
            train_loader = tqdm(self.train_loader_4, ncols=0, desc='Train')
        elif cross_val_num == 5:
            train_loader = tqdm(self.train_loader_5, ncols=0, desc='Train')

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            #여기도 문제임.
            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')

        return train_loss, train_acc

    def evaluate(self, cross_val_num):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            if cross_val_num == 1:
                valid_loader = tqdm(self.train_loader_1, desc='Validate', ncols=0)
            elif cross_val_num == 2:
                valid_loader = tqdm(self.train_loader_2, desc='Validate', ncols=0)
            elif cross_val_num == 3:
                valid_loader = tqdm(self.train_loader_3, desc='Validate', ncols=0)
            elif cross_val_num == 4:
                valid_loader = tqdm(self.train_loader_4, desc='Validate', ncols=0)
            elif cross_val_num == 5:
                valid_loader = tqdm(self.train_loader_5, desc='Validate', ncols=0)
            for x, y in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = F.cross_entropy(output, y)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(output, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')

        return valid_loss, valid_acc

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
