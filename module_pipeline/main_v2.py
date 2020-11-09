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

def extract_frame(video_path):
    pretrained_model = models.alexnet(pretrained=True)
    pretrained_model.eval()
    print(video_path)
    video = cv2.VideoCapture(video_path)
    count = 0
    num = -1
    
    videoframe = video.get(cv2.CAP_PROP_POS_FRAMES)
    videocount = video.get(cv2.CAP_PROP_FRAME_COUNT)

    while video.isOpened():
        #ret 은 프레임이 존재하지 않을때 T/F 반환
        #frame 은 프레임 반환
        num += 1
        ret, frame = video.read()
        if ret == False:
            print("Frame doesn't Exist")
            break
        if num % 15 == 0:
            if(videoframe == videocount):
                video.open(filename)
            

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

            try:

                #face_location 은 얼굴 위치(행렬에서)
                face_location = face_recognition.face_locations(frame) #cnn 모델 사용 X. 원할시, model='cnn' 입력(GPU 없는 기기에서는 실행되지 않음)
                top, right, bottom, left = face_location[0]
                print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

                #face_location 찾아서 파일로 저장
                if 0 <= top-10 and bottom < len(frame) and 0 <= left and right+10 < len(frame[0]):
                    face_image = Image.fromarray(frame[top-10:bottom+10, left-10:right+10])
                    trans = torchvision.transforms.ToTensor()
                    face_image = trans(face_image)
                    c, h, w = face_image.shape
                    out = pretrained_model(face_image.reshape(1, c, h, w))
                    count += 1
                else:
                    continue

            except Exception as e:
                print("Exception Occurs.", e)
                pass

    video.release()
    cv2.destroyAllWindows()
    return out, count


def main():
    video_path = "./000_M101.mp4"
    result, cnt = extract_frame(video_path)
    print("cnt : {}".format(cnt))
    print("result : {}".format(result))

if __name__ == "__main__":
    main()