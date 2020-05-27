import os
from PIL import Image
from skimage import io, transform
import cv2
import numpy as np
import mlconfig
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class Expand(object):

    def __call__(self, t):
        return t.expand(3, t.size(1), t.size(2))


class DEEPFAKE_train_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        dir_0 = "./configs/train/0/"
        dir_1 = "./configs/train/1/"

        file_list_0 = os.listdir(dir_0)
        file_list_1 = os.listdir(dir_1)
        
        self.data_dict = {}

        for i in range(len(file_list_0)):
            self.data_dict[i] = (dir_0+file_list_0[i], 0)

        for i in range(len(file_list_1)):
            self.data_dict[len(file_list_0)+i] = (dir_1+file_list_1[i], 1)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_path, label = self.data_dict[idx]
        #image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = Image.open(img_path)
        
        #image = np.swapaxes(image, 0, 2)
        #image = np.swapaxes(image, 1, 2)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            Expand(),
        ])
        image = transform(image)
        return (image, label)

@mlconfig.register
class DEEPFAKE_train_DataLoader(DataLoader):
    def __init__(self, batch_size: int, **kwargs):
        dataset = DEEPFAKE_train_Dataset()

        super(DEEPFAKE_train_DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, **kwargs)
