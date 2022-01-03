import detectron2
import numpy as np
import os, json, cv2

from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances,load_coco_json
from detectron2.utils.logger import setup_logger



from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
setup_logger()

from config_segment import sconfig

class RCNN_train:
    
    def __init__(self,num_classes,session_name,image_folder,annotation_file,resume=False):
        self.batch_size = sconfig['batch_size']
        self.num_classes = num_classes
        self.session_name = session_name
        self.annotation_file = annotation_file
        self.image_folder = image_folder
        self.resume= resume
        self.define_model()
        self.prepare_dataset()
        self.prepare_configs()
    
    def define_model(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        self.cfg.DATALOADER.NUM_WORKERS = sconfig['DATALOADER.NUM_WORKERS']
        if self.resume:
            self.cfg.MODEL.WEIGHTS = self.cfg.OUTPUT_DIR
        else:
            self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
        self.prepare_configs()
    
    def prepare_dataset(self):
        DatasetCatalog.clear()
        register_coco_instances("pollen_dataset", {}, self.annotation_file, self.image_folder)
        # self.cfg.DATASETS.TRAIN =  load_coco_json("/home/laanta/sagun/segmentation/train_pollen_dec28.json", "/home/laanta/sagun/segmentation/data/images", dataset_name='pollen_dataset',)
        self.cfg.DATASETS.TRAIN = ("pollen_dataset",)
        self.cfg.DATASETS.TEST = () 
        self.dataset_dicts = DatasetCatalog.get("pollen_dataset")

    def prepare_configs(self):
         # no metrics implemented for this dataset
        cwd = os.getcwd()
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg.SOLVER.BASE_LR = sconfig['LR']
        self.cfg.SOLVER.MAX_ITER = (
            sconfig['MAX_ITER']
        )  
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            sconfig['MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE']
        )  
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # 3 classes (data, fig, hazelnut)
        self.cfg.OUTPUT_DIR =  os.path.join(cwd,'segmentation_sessions',self.session_name)
        
    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=self.resume)
        trainer.train()
                    

# pollen_metadata = MetadataCatalog.get("pollen_dataset")
if __name__ == '__main__':
    obj = RCNN_train(4,'test',image_folder='full_data/images',annotation_file='train_pollen_jan1.json', \
                resume=True)
    obj.train()




