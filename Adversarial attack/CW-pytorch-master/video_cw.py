# python -m Deept-ADIS.CW-pytorch-master.CW

import numpy as np
import json
import cv2
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models import *

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])



for images, labels in normal_loader:
    
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    
    _, pre = torch.max(outputs.data, 1)
    
    total += 1
    correct += (pre == labels).sum()
    

def cw_l2_attack(model, images, labels, targeted=True, c=1e-4, kappa=0, max_iter=100, learning_rate=0.01) :

    images = images.to(device)     
    labels = labels.to(device)

    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        
        return torch.clamp(i-j, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

model.eval()

correct = 0
total = 0

for images, labels in normal_loader:
    
    images = cw_l2_attack(model, images, labels, targeted=False, c=0.1)
    labels = labels.to(device)
    outputs = model(images)
    
    _, pre = torch.max(outputs.data, 1)

    correct += (pre == labels).sum()
    
    for i in range(images.shape[0]):
        total += 1
        print(total)
        save_image(images.cpu().data[i], "./data/cw/"+normal_data.classes[labels.cpu()[i]]+"/"+str(total)+"_"+normal_data.classes[pre[i]]+".jpg")
    
print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


