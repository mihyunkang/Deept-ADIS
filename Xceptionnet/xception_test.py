import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pretrainedmodels
from pretrainedmodels.models import *
from datasets.trainer import Trainer 
from datasets.metrics import *
from datasets.deepfake_test import DEEPFAKE_test_DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--arch', type=str, default='efficientnet_b0')
    #parser.add_argument('--arch', type=str, default="./checkpoint/")
    #parser.add_argument('-r', '--root', type=str, default='./configs/')
    parser.add_argument('--batch-size', type=int, default=64)#128
    parser.add_argument('-w', '--weight', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    #parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()


def evaluate(testmodel, valid_loader, device):

    PATH = "./checkpoint/xception/best.pth"
    checkpoint = torch.load(PATH)
    model = testmodel # 이게 되려나...?
    model.load_state_dict(checkpoint['model'])
    model.eval()

    valid_loss = Average()
    valid_acc = Accuracy()

    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc='Validate', ncols=0)
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x) #모델 출력
            loss = F.cross_entropy(output, y)

            valid_loss.update(loss.item(), number=x.size(0))
            valid_acc.update(output, y)

            valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')

    return valid_loss, valid_acc


if __name__ == '__main__':

    print("[*] evaluate running ... ")
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = "./checkpoint/xception/best.pth"
    model = xception() #모델이 이거 하나면 되는건가?5
    model.to(device)

    #image_size = params[args.arch][2]
    #print("image_size", image_size)
    #valid_loader = DEEPFAKE_test_DataLoader(args.root, image_size, False, shuffle = True, args.batch_size, num_workers=args.num_workers)
    valid_loader = DEEPFAKE_test_DataLoader(batch_size = args.batch_size, shuffle=True)

    evaluate(model, valid_loader, device)

    print("-------------- Test finished --------------\n")
