import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import cv2
import numpy as np
import face_recognition
import re
import os
import torchvision
from torchvision import models
from PIL import Image
from torchvision import transforms

import pretrainedmodels
from pretrainedmodels.models import *

class Expand(object):
    def __call__(self, t):
        return t.expand(3, t.size(1), t.size(2))

def extract_frame(video_path):
    trans_PIL = transforms.ToPILImage()
    transform = transforms.Compose([
            #transforms.Resize((256,256)),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomCrop((20,20)), 
            #transforms.RandomRotation(90, expand=True),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #mean, std
            Expand(),
        ])
        #이미지를 다시 저장..?!
        #filpy = transforms.RandomVerticalFlip(p=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = "./checkpoint/xception/best.pth"
    checkpoint = torch.load(PATH)
    model = xception(1000)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(video_path)
    cap = cv2.VideoCapture(video_path)
    count = 0
    num = -1
    
    videoframe = cap.get(cv2.CAP_PROP_POS_FRAMES)
    videocount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, frame = cap.read()
    w, h, c = frame.shape
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    out = cv2.VideoWriter('2.mp4',fourcc , 20, (frame_width, frame_height))

    while cap.isOpened():
        #ret 은 프레임이 존재하지 않을때 T/F 반환
        #frame 은 프레임 반환
        num += 1
        ret, frame = cap.read()
        if ret == False:
            print("Frame doesn't Exist")
            break
        
        if(videoframe == videocount):
            cap.open(filename)
            

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

        try:

            #face_location 은 얼굴 위치(행렬에서)
            face_location = face_recognition.face_locations(frame) #cnn 모델 사용 X. 원할시, model='cnn' 입력(GPU 없는 기기에서는 실행되지 않음)
            top, right, bottom, left = face_location[0]
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            #for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #face_location 찾아서 파일로 저장
            out.write(frame) # 동영상 프레임 마다 쓰는 부분
            if 0 <= top-10 and bottom < len(frame) and 0 <= left and right+10 < len(frame[0]):
                face_image_npy = Image.fromarray(frame[top-10:bottom+10, left-10:right+10])
                trans = torchvision.transforms.ToTensor()
                face_image = trans(face_image_npy)
                c, h, w = face_image.shape
                # result = model(transform(trans(face_image.reshape(1, c, h, w))))
                result = model(transform(face_image_npy))
                count += 1
            else:
                continue

        except Exception as e:
            print("Exception Occurs.", e)
            pass

    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    return result, count


def main():
    video_path = "./000_M101.mp4"
    result, cnt = extract_frame(video_path)
    print("cnt : {}".format(cnt))
    print("result : {}".format(result))

if __name__ == "__main__":
    main()