"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import copy
import cv2
import numpy as np
#imread용
import matplotlib.pyplot as plt
from PIL import Image

import skimage

import torch
from torch.autograd import Variable
from torchvision import models

#from efficientnet_pytorch import EfficientNet

#도전1... efficientnet.py 모듈 불러오기...
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from efficientnet.efficientnet.models import efficientnet

#여기서, efficientnet\efficientnet\datasets\deepfake_test.py 를 위해 pip install mlconfig 해줌.

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    im_as_arr = np.float32(cv2im)
    a, _, _ = im_as_arr.shape
    print(a)
    if a == 3:
        im_as_arr = im_as_arr.transpose(1, 2, 0)
    im_as_arr = cv2.resize(im_as_arr, (224, 224))
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    #im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    print("---------ss--")
    print(im_as_arr.shape)
    # Normalize the channels
    cnt = 0
    while cnt <3:
        for channel, _ in enumerate(im_as_arr):
            #print(channel)
            im_as_arr[channel] /= 255
            #값 전체에 뺀다는 얘기 같은데... im_as_arr 는 지금 큰 배열임.
            im_as_arr[channel] -= mean[cnt] #np.mean(im_as_arr[channel]) 로 하면 1x3x3 dimension 이 되어버림...
            im_as_arr[channel] /= std[cnt]
        cnt += 1
            #print(im_as_arr.shape)
    
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    #print(im_as_ten)
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)#이 값이 뭔가 중요한 역할을 하는듯,,,
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    #print(im_as_var)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    #recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


def get_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    # (2) 여기 바꿔야함...!
    example_list = [["../input_images/2.JPG",1]]
    selected_example = example_index
    img_path = example_list[selected_example][0]
    target_class = example_list[selected_example][1] 
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    
    # Read image
    original_image = skimage.io.imread(img_path)
    print(original_image.shape)
    # Process image
    prep_img = preprocess_image(original_image)[0]
    #prep_img = [prep_img]
    print("process image ", prep_img.shape)
    
    # Define model (1) 여기 바꾸고
    #pretrained_model = models.alexnet(pretrained=True)
    PATH = "../../efficientnet/checkpoint/best_per100_val_92.pth"
    pretrained_model = efficientnet.EfficientNet()

    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)

#print(sys.path)
#C:\deept\workspace\Deept-ADIS\pytorch-cnn-adversarial-attacks-master\input_images\snake.JPEG'
