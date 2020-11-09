import cv2
import face_recognition
from random import * 
import imutils
import torch
import numpy as np

import numpy as np
import json
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

PATH = "D:/Deept-ADIS/pretrained-models.pytorch-master/checkpoint/xception/best.pth"
model = xception().to(device)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model'])
model.eval()

correct = 0
total = 0

if __name__ == '__main__':
   
    # capture frames from a video
    cap = cv2.VideoCapture(r'fake.mp4')
    labels = 1
    #cap.set(cv2.CAP_PROP_FPS, int(60))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Initialize variables
    face_locations = []
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('fake_cw.avi',fourcc, 20.0, (frame_width, frame_height))
    #out = cv2.VideoWriter('1_out.mp4',fourcc , 3, (frame_width, frame_height))
    while(cap.isOpened()) :
        # Grab a single frame of video
        ret, frame = cap.read()
    
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            face_image = frame[top:bottom, left:right]
            np.resize(face_image, (256, 256))           
            face_cw = cw_l2_attack(m, odel, face_image, labels, targeted=False, c=0.1)
            frame[top:bottom, left:right] = face_cw
    
        
        out.write(frame)
        # Wait for Enter key to stop
        if cv2.waitKey(25) == 13:
            break

        
        
    # Release everything if job is finished
    out.release() 
    cap.release()
    cv2.destroyAllWindows()