"""
Created on Fri Dec 16 01:24:11 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None

from misc_functions import preprocess_image, recreate_image, get_params
from efficientnet.efficientnet.models import efficientnet
from efficientnet.efficientnet.optim.rmsprop import TFRMSprop


class FastGradientSignTargeted():
    """
        Fast gradient sign untargeted adversarial attack, maximizes the target class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        PATH = "../../efficientnet/checkpoint/best_per100_val_92.pth"
        self.checkpoint = torch.load(PATH)
        self.model = model
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self, original_image, org_class, target_class):
        # I honestly dont know a better way to create a variable with specific value
        # Targeting the specific class
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])))
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(10):
            print('Iteration:', str(i))
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            print(out.dtype)
            print(im_label_as_var.dtype)
            pred_loss = F.cross_entropy(out, im_label_as_var.long())
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add noise to processed image
            processed_image.data = processed_image.data - adv_noise
            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            confirmation_out = self.model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            # Check if the prediction is different than the original
            if confirmation_prediction == target_class:
                print('Original image was predicted as:', org_class,
                      'with adversarial noise converted to:', confirmation_prediction,
                      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                original_image = cv2.resize(original_image, (224, 224))
                recreated_image = recreated_image.transpose(1, 2, 0)
                noise_image = original_image - recreated_image
                cv2.imwrite('../generated/targeted_adv_noise_from_' + str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                cv2.imwrite('../generated/targeted_adv_img_from_' + str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', recreated_image)
                break

        return 1


if __name__ == '__main__':
    print("FGSM targeted 실행")
    target_example = 0  # Apple
    (original_image, prep_img, org_class, _, pretrained_model) =\
        get_params(target_example)
    target_class = 0 # Mud turtle

    FGS_untargeted = FastGradientSignTargeted(pretrained_model, 0.02)
    FGS_untargeted.generate(original_image, org_class, target_class)
