import os
from PIL import Image
from skimage import io, transform
from skimage.util import random_noise
import cv2
import numpy as np
import mlconfig
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
import random

class Expand(object):
    def __call__(self, t):
        return t.expand(3, t.size(1), t.size(2))


class DEEPFAKE_train_Dataset(Dataset):
    """Face Landmarks dataset."""
    #전체 데이터 : X, fake/real(라벨) : y 로 딕셔너리에 넣어줌.
    #key 와 value로 떼올 수 있지 않을까...?

    def __init__(self, num):
        #딥페이크 프레임 데이터 로드.
        #매번 테스트/트레이닝 할 때마다 바꿔주는 경로
        dir_1 = 'D:/dataset/train/fake/' #"/home/deept/data/full_data/fake/" #전체 fake 이미지
        dir_0 = 'D:/dataset/train/real/'#"/home/deept/data/full_data/real/" #전체 real 이미지

        file_list_0 = os.listdir(dir_0)
        file_list_1 = os.listdir(dir_1) #무작위로 섞음... 너무 랜덤인가 싶기도...?
        #random.shuffle(file_list_0)
        #random.shuffle(file_list_1)
        
        self.cross_val_num = num #0~4 -> 5-cross validation
        self.data_dict = {}
        self.max_len  = max(len(file_list_0), len(file_list_1))
        for i in range(self.max_len*2):
            if(i%2==0): #data_dict 의 짝수번 인덱스는 real data 저장
                self.data_dict[i] = (dir_0+file_list_0[(i//2)%len(file_list_0)], 0)
            else: #홀수번 인덱스에는 fake data 저장
                self.data_dict[i] = (dir_1+file_list_1[(i//2)%len(file_list_1)], 1)
        #self.data_dict = self.data_dict[((self.max_len*2)//5)*self.cross_val_num:((self.max_len*2)//5)*(self.cross_val_num+1)]
        self.data_dict = dict(list(self.data_dict.items())[((self.max_len*2)//5)*self.cross_val_num:((self.max_len*2)//5)*(self.cross_val_num+1)])


    def __len__(self):
        return (self.max_len)*2

    def __getitem__(self, idx):
        img_path, label = self.data_dict[idx]
        #image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = Image.open(img_path)
        ## augmentation~~
        transform = transforms.Compose([
            #transforms.RandomVerticalFlip(),
            #transforms.RandomCrop((20,20)), 
            #transforms.RandomRotation(90, expand=True),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #mean, std
            Expand(),
        ])
        #이미지를 다시 저장..?!
        image = transform(image)
        return (image, label) #X, y -> cross validation 할 수 있을듯!

@mlconfig.register
class DEEPFAKE_train_DataLoader(DataLoader):
    def __init__(self, cross_val_num, batch_size: int, **kwargs):
        dataset = DEEPFAKE_train_Dataset(cross_val_num) #cross_val_num 이 Dataset 안에 들어간다고..?

        super(DEEPFAKE_train_DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, **kwargs)
