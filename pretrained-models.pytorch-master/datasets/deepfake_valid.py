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
        self.max_len  = max(len(file_list_0), len(file_list_1))
        for i in range(self.max_len*2):
            if(i%2==0): #data_dict 의 짝수번 인덱스는 real data 저장
                self.data_dict[i] = (dir_0+file_list_0[(i//2)%len(file_list_0)], 0)
            else: #홀수번 인덱스에는 fake data 저장
                self.data_dict[i] = (dir_1+file_list_1[(i//2)%len(file_list_1)], 1)


    def __len__(self):
        #return len(self.data_dict)
        return (self.max_len)*2

    def __getitem__(self, idx):
        img_path, label = self.data_dict[idx]
        #image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #mean, std
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
