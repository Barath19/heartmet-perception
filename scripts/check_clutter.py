from typing import List

import cv2
import numpy as np
from ultralytics import FastSAM
from ultralytics.models import YOLO
from ultralytics.models.fastsam import FastSAMPrompt

class CheckClutter:

    def __init__(self, model_path='FastSAM-s.pt', bounding_boxes=None, prompt=None):
        self.model = FastSAM(model_path)  # or FastSAM-x.pt
        self.bounding_boxes = bounding_boxes
        #self.prompt = prompt.strip().split(' ')
        self.prompt_process = None

    def run_everything(self, img):
        # Run inference on an image
        self.everything_results = self.model(img, device=0, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        # Prepare a Prompt Process object
        self.prompt_process = FastSAMPrompt(img, self.everything_results, device=0)


    def run_with_prompt(self, prompt:str) -> np.array:
        return self.prompt_process.text_prompt(text=prompt)
    
    def get_combined_masks(self, masks:List) -> np.array:
        return np.logical_or.reduce(masks)
    
    def get_binary_img(self, mask:np.array) -> np.array:
        return np.uint8(mask[0]*255)





# Everything prompt
#ann = prompt_process.everything_prompt()

# Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
#ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])



if __name__ == '__main__':
    img_file = "scripts/test.jpg"
    img = cv2.imread(img_file)

    clutter = CheckClutter()
    clutter.run_everything(img)

    objects = ['toothbrush', 'cup', 'joystick', 'cube']
    # Text prompt
    #masks = np.zeros(img.shape, dtype=bool)
    masks = {}
    all_mask = []
    for object in objects:
        mask = clutter.run_with_prompt(object)
        masks[object] = mask
        all_mask.append(mask)

    #ann = prompt_process.text_prompt(text='a cup')

    #for mask in masks:
    final_mask = clutter.get_combined_masks(all_mask)

    cv2.imshow('clutter_mask', clutter.get_binary_img(final_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
