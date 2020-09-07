import os
import cv2
import mlconfig
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class Expand(object):

    def __call__(self, t):
        return t.expand(3, t.size(1), t.size(2))


class DEEPFAKE_val_Dataset(Dataset):
    def __init__(self):
        dir_1 = "D:/full_data/val/fake/"
        dir_0 = "D:/full_data/val/real/"        

        file_list_0 = os.listdir(dir_0)
        file_list_1 = os.listdir(dir_1)
        
        self.data_dict = {}
        self.len  = len(file_list_0)+len(file_list_1)
        for i in range(self.len):
            if(i<len(file_list_0)): #data_dict 의 짝수번 인덱스는 real data 저장
                self.data_dict[i] = (dir_0+file_list_0[i], 0)
            else: #홀수번 인덱스에는 fake data 저장
                self.data_dict[i] = (dir_1+file_list_1[i-len(file_list_0)], 1)


    def __len__(self):
        #return len(self.data_dict)
        return self.len 
        
    def __getitem__(self, idx):
        img_path, label = self.data_dict[idx]
        #image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), #이건 숫자가 왜 이럴까?
            Expand(),
        ])
        image = transform(image)
        #print(image, label)
        return (image, label)

@mlconfig.register
class DEEPFAKE_val_DataLoader(DataLoader):

    def __init__(self, batch_size: int, **kwargs):

        dataset = DEEPFAKE_val_Dataset()

        super(DEEPFAKE_val_DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, **kwargs)
