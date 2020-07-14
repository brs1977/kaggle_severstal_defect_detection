import argparse
import numpy as np
import cv2
import torch.nn as nn
import torch 
import os
from catalyst.dl import utils
import albumentations as A
import ttach as tta
from albumentations.pytorch import ToTensorV2 as ToTensor
from mlcomp.contrib.transform.albumentations import ChannelTranspose
import segmentation_models_pytorch as smp
from skimage import morphology

thresholds = [.45, .45, .45, .45] #[0.7, 0.7, 0.6, 0.6]
min_area = [600, 600, 900, 2000]
regions_size = 512
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

class Predictor:
    def __init__(self, model_path):
        self.augmentation = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensor()])

        tta_transforms = tta.Compose(
            [
                tta.Rotate90([0]), # NoOp
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ])        
        self.m = nn.Sigmoid()        
        self.model = self.load_model_traced(model_path)
        self.model = tta.SegmentationTTAWrapper(self.model, tta_transforms, merge_mode="mean")


        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

    def load_model_traced(self, model_path):
        return utils.trace.load_traced_model(model_path).cuda()

    def load_model_checkpoint(self, model_path):    
        model = smp.FPN(encoder_name="timm-efficientnet-b3", classes=4)
        checkpoint = utils.load_checkpoint(model_path)
        utils.unpack_checkpoint(checkpoint, model=model)
        return model.cuda()

    def predict(self, image):
        augmented = self.augmentation(image=image)
        inputs = augmented['image']
        inputs = inputs.unsqueeze(0).to(self.device) 
        print(inputs.shape)
        logits = self.m(self.model(inputs)).cpu().numpy()[0]

        masks = logits.transpose(1,2,0)
                
        # Image postprocessing
        for i in range(4):            
            mask = masks[...,i]    
            #порог предсказания вероятности
            mask = (mask>thresholds[i]) 
            
            #порог по минимальному региону
            mask = remove_small_regions(mask, regions_size)
            if mask.sum() < min_area[i]:
                mask = np.zeros(mask.shape, dtype=mask.dtype)
            mask = mask.astype(np.uint8)
            contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for j in range(0, len(contours)):
                cv2.polylines(image, contours[j], True, palet[i], 2)                
                
#             show_examples(name="", image=image, mask=mask)
                
        return image
   

parser = argparse.ArgumentParser(description='Defect mask predictor')
parser.add_argument('-i', action='store', required=True, dest='image_path', help='Input image file path')
parser.add_argument('-o', action='store', dest='output_path', help='Output path', default='output')
parser.add_argument('-m', action='store', dest='model_path', help='Model path', default='logs/fpn_timm-effb3/trace/traced-best-forward.pth')

if __name__ == "__main__":    
    # args = parser.parse_args(['-i', 'data/test_images/0af0c1a38.jpg'])
    args = parser.parse_args()

    predictor = Predictor(args.model_path)

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = predictor.predict(image)
    out_file_name = os.path.join(args.output_path, os.path.basename(args.image_path))
    cv2.imwrite(out_file_name, image)

    