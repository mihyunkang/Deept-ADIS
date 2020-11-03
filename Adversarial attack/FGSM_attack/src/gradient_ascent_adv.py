"""
Created on Thu Oct 29 14:09:01 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import torch

from torch.optim import SGD
from torchvision import models
from torch.nn import functional

#여기구나~~!!
from misc_functions import preprocess_image, recreate_image, get_params
from efficientnet.efficientnet.models import efficientnet
from efficientnet.efficientnet.optim.rmsprop import TFRMSprop

class DisguisedFoolingSampleGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent, breaks as soon as
        the target prediction confidence is captured
    """
    def __init__(self, model, initial_image, target_class, minimum_confidence):
        PATH = "../../efficientnet/checkpoint/best_per100_val_92.pth"
        self.checkpoint = torch.load(PATH)
        self.model = model
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()
        #target_class 는 fooling class!
        self.target_class = target_class
        self.minimum_confidence = minimum_confidence
        # Generate a random image
        self.initial_image = initial_image
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self):
        for i in range(1, 500):
            # Process image and return variable
            #initial_image 가 originial image 임....
            self.processed_image = preprocess_image(self.initial_image) #224x224 크기의 이미지...
            
            # Define optimizer for the image
            optimizer = TFRMSprop(self.model.parameters())
            
            optimizer.load_state_dict(self.checkpoint['optimizer'])
            # Forward
            #print(self.model)
            output = self.model(self.processed_image) #dim=2 추가함.(안됨)
            #여기서 model 은 get_params 로 받은 efficientnet임.
            # Get confidence from softmax
            #print(self.target_class) -> 1000개임!
            #print(functional.softmax(output).data.numpy())
            #[0][self.target_class]
            target_confidence = functional.softmax(output).data.numpy()[0][1]
            if target_confidence > self.minimum_confidence:
                # Reading the raw image and pushing it through model to see the prediction
                # this is needed because the format of preprocessed image is float and when
                # it is written back to file it is converted to uint8, so there is a chance that
                # there are some losses while writing
                confirmation_image = cv2.imread('../generated/ga_adv_class_' +
                                                str(self.target_class) + '.jpg', 1)
                print(confirmation_image.shape)
                # Preprocess image
                confirmation_processed_image = preprocess_image(confirmation_image)
                # Get prediction
                confirmation_output = self.model(confirmation_processed_image)
                # Get confidence
                softmax_confirmation = \
                    functional.softmax(confirmation_output)[0][self.target_class].data.numpy()[0]
                if softmax_confirmation > self.minimum_confidence:
                    print('Generated disguised fooling image with', "{0:.2f}".format(softmax_confirmation),
                          'confidence at', str(i) + 'th iteration.')
                    break
            # Target specific class
            #print(self.target_class) -> 1000 임....
            print("output ", output)
            class_loss = -output[0, 1]#self.target_class]
            #class_loss = -output[0,1], -output[0, 0] 은 실행되는데, 원래의 self.target_class는 안됨... ㅠㅠㅠ
            print('Iteration:', str(i), 'Target confidence', "{0:.4f}".format(target_confidence))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.initial_image = recreate_image(self.processed_image)
            # Save image
            self.initial_image = self.initial_image.transpose(1, 2, 0) 
            cv2.imwrite('../generated/ga_adv_class_' + str(self.target_class) + '.jpg',
                        self.initial_image)
        return confirmation_image


if __name__ == '__main__':
    print("Gradient_ascent_adv.py 실행중")
    target_example =  0 
    (original_image, prep_img, _, _, pretrained_model) =\
        get_params(target_example)

    fooling_target_class = 0  # Abacus -> 398 에서 바꿔줌.
    min_confidence = 0.99
    fool = DisguisedFoolingSampleGeneration(pretrained_model,
                                            original_image,
                                            fooling_target_class,
                                            min_confidence)
    generated_image = fool.generate()
