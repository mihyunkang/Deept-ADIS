import argparse
import mlconfig
import torch
from torch import distributed, nn, optim
import pretrainedmodels
from pretrainedmodels.models import *
from datasets.trainer import Trainer 

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--config', type=str, default='./configs/train/')
    parser.add_argument('--resume', type=str, default=None)
    #parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data-parallel', action='store_true')

    # distributed
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    return parser.parse_args()


def init_process(backend, init_method, world_size, rank):
    distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )


def main():
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if args.world_size > 1:
        init_process(args.backend, args.init_method, args.world_size, args.rank)
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = inceptionv4()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    #optimizer = TFRMSprop(model.parameters())
    lmbda = lambda epoch: 0.95
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    #scheduler = torch.optim.lr_scheduler._LRScheduler(optimizer)
    trainerclass = Trainer(model, optimizer, scheduler, device, 100, "./checkpoint/")

    if args.resume is not None:
        trainerclass.resume(args.resume)

    trainerclass.fit()


if __name__ == "__main__":
    main()
    print("------------------ finished. ----------------\n")