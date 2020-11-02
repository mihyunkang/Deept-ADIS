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

from misc_functions import preprocess_image, recreate_image, get_params


class FastGradientSignTargeted():
    """
        Fast gradient sign untargeted adversarial attack, maximizes the target class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        PATH = "../../Xceptionnet/checkpoint/xception/best.pth"
        self.checkpoint = torch.load(PATH)
        self.model = model
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('../generated/targeted'):
            os.makedirs('../generated/targeted')

    def generate(self, original_image, org_class, target_class, image_path):
        # I honestly dont know a better way to create a variable with specific value
        # Targeting the specific class
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])))
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image) #original image를 var로 변환한것.
        # Start iteration
        for i in range(10):
            print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image) #model을 거친 값.
            #print(out.dtype) #float
            #print(im_label_as_var.dtype) #int
            # Calculate CE loss
            pred_loss = ce_loss(out, im_label_as_var.long()) #type error 때문에 오류 발생했음. 
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data) #노이즈 만들기.
            # Add noise to processed image
            processed_image.data = processed_image.data - adv_noise #기존의 이미지에 노이즈를 뺌. 다시 생성한 이미지~~

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image) 
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image) #새로 생성한 이미지를 var 형태로 변환.
            # Forward pass
            confirmation_out = self.model(prep_confirmation_image) #model로 evaluate 한 값
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1) #같은 모델로 돌렸을 때, 결과값(=예측값?) 좀 걸리는게,, 0과 1 계속 왔다갔다함.
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0] #얼마나 일치하는지..softmax 서치필요
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0] #0과 1로 표현됨.... -> fake or real 이었다!
            # Check if the prediction is different than the original
            if confirmation_prediction == target_class:
                print('Original image was predicted as:', org_class,
                      'with adversarial noise converted to:', confirmation_prediction,
                      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                original_image = cv2.resize(original_image, (224,224))
                recreate_img = recreated_image.transpose(0, 1, 2)
                noise_image = original_image - recreate_img #고작 이게 노이즈..?
                #cv2.imwrite('../generated/targeted/noise_from_' + image_path+'_'+str(org_class) + '_to_' +
                #            str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                cv2.imwrite('../generated/targeted/adv_img_from_' + image_path+ '_'+str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', recreate_img)

                #FGSM 생성한 이미지의 Xception 모델 결과값(Fake/Real).
                #deepfake_output = self.model(torch.Tensor(recreate_img))
                deepfake_output = self.model(processed_image) #torch.from_numpy(np.flip(recreate_img, axis=0).copy())
                _, FGSM_result = deepfake_output.data.max(1)
                print("FGSM result : ", FGSM_result.numpy()[0])
                break
        return 1

#targeted 말구 untargeted 로 하면 안되려나..?!
#내가 생각한 생성방법이랑 너무 다른듯... 모든 이미지에 해당하는 AE 가 생성되어야 하는거 아닌가?
if __name__ == '__main__':
    print("------------ Targeted FGSM 실행 ----------")
    folder_pth = 'D:/full_data/test/real'
    target_class = 1 #fake 라 1임.
    for i in range(len(os.listdir(folder_pth))):
        #전체 파일 리스트가 for문 동안 계속 왔다갔다하면 비효율적인데...
        (original_image, prep_img, org_class, _, pretrained_model, img_path) = get_params(i)
        FGS_untargeted = FastGradientSignTargeted(pretrained_model, 0.02)
        FGS_untargeted.generate(original_image, org_class, target_class, img_path)


    
    #FGSM -> model 에 넣어서 결과 출력

