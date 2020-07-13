import numpy as np
import cv2
import torch.nn as nn
import torch 
import os
from catalyst.dl import utils
import albumentations as A
from mlcomp.contrib.transform.albumentations import ChannelTranspose
import segmentation_models_pytorch as smp


class Predictor:
    def __init__(self, model_path, image_size):
        self.augmentation = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ChannelTranspose()])

        self.m = nn.Sigmoid()        
        self.model = self.load_model(model_path)

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

    def load_model(self, model_path):
        model = smp.FPN(encoder_name="timm-efficientnet-b3", classes=4)
        checkpoint = utils.load_checkpoint(f"{model_path}/checkpoints/best.pth")
        utils.unpack_checkpoint(checkpoint, model=model)
        return model.cuda()

    def predict(self, image,threshold):
        augmented = self.augmentation(image=image)
        inputs = augmented['image']
        inputs = inputs.unsqueeze(0).to(self.device) 
        output = self.m(self.model(inputs)[0][0]).cpu().numpy()
        probability = (output > threshold).astype(np.uint8)
        return probability