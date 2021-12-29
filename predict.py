from detectron2.utils.visualizer import ColorMode

import detectron2
import matplotlib.pyplot as plt
# import some common libraries
import numpy as np
import os, json, cv2, random

import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from imantics import Polygons, Mask


class Predictor:
    
    def __init__(self,weights_path):
        self.weights_path = weights_path
        self.num_classes =4
        self.define_model()
        pass

    def define_model(self):
        self.cfg = get_cfg()
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = self.weights_path  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold
        
        self.prepare_configs()
        self.predictor = DefaultPredictor(self.cfg)
    
    def prepare_configs(self):
         # no metrics implemented for this dataset
        cwd = os.getcwd()
        
        self.cfg.SOLVER.MAX_ITER = (
            300
        )  # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128
        )  # faster,  dataset
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # 3 classes (data, fig, hazelnut)
        
    def predict(self,img_path,plot=False):
        
        im = cv2.imread(img_path) 
        outputs = self.predictor(im)
        
        
        pred_classes = outputs['instances'].pred_classes.to("cpu").numpy()
        pred_boxes = outputs['instances'].pred_boxes.to("cpu")
        scores = outputs['instances'].scores.to("cpu").numpy()
        mask_array = outputs['instances'].pred_masks.to("cpu").numpy()

        pol_points=[]
        output_boxes = []
        for i in mask_array:
            polygons = Mask(i).polygons()
            pol_points.append(polygons.points)

        for box,label in zip(pred_boxes,pred_classes):
            box = box.cpu().detach().numpy()
            
            # label = map_label[str(label)]
            x1,y1,x2,y2 = box.astype('int')
            output_boxes.append([label,x1,y1,x2,y2])

        
        if plot:
            self.pol_plot(im,pol_points,output_boxes,img_path)
        return pol_points,output_boxes
    
    def pol_plot(self,im,pol_points,boxes,img_path):
        
        fig = plt.figure(figsize=(20,10))
        color = (255, 0, 0)
        thickness = 1
        isClosed = False

        map_label ={'1':'POL',
            '0':'ANOM',
            '2':'OTHPAR',
            '3':'MOL'}
        
        map_color ={'POL':(255, 0, 0),
            'ANOM':(255, 255, 0),
            'OTHPAR':(255, 0, 255),
            'MOL':(0, 0, 255)}

        for points,box in zip(pol_points,boxes):
            
            
            label,x1,y1,x2,y2 = box
            label = map_label[str(label)]

            color = map_color[str(label)]
            im =cv2.polylines(im, points, isClosed, color, thickness)
            # cv2.putText(im,label, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            pl= cv2.rectangle(im, (x1,y1), (x2,y2),(0, 255, 0), thickness=1)
            
    
        
        img_name = img_path.split('/')[-1]
       
        cv2.imwrite('out_predict/'+img_name,im)
        

weights = '/home/laanta/sagun/segmentation/output/model_final.pth'
img_path = '/home/laanta/sagun/segmentation/data/images/0ed9aa9a-1347-4188-b396-77aa0dd90749.png'
obj= Predictor(weights)

for img in os.listdir('data/images'):
    print(img)
    img_path = os.path.join('data/images/',img)
    out =obj.predict(img_path,plot=True)
