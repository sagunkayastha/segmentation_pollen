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

class RCNN_train:
    
    def __init__(self,batch_size,num_classes,session_name):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.session_name = session_name
        self.define_model()
        self.prepare_dataset()
        self.prepare_configs()
        
    
    def define_model(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
        self.prepare_configs()
    
    def prepare_dataset(self):
        DatasetCatalog.clear()
        register_coco_instances("pollen_dataset", {}, "/home/laanta/sagun/segmentation/train_pollen_dec28.json", "/home/laanta/sagun/segmentation/data/images")
        # self.cfg.DATASETS.TRAIN =  load_coco_json("/home/laanta/sagun/segmentation/train_pollen_dec28.json", "/home/laanta/sagun/segmentation/data/images", dataset_name='pollen_dataset',)
        self.cfg.DATASETS.TRAIN = ("pollen_dataset",)
        self.cfg.DATASETS.TEST = () 
        self.dataset_dicts = DatasetCatalog.get("pollen_dataset")

    def prepare_configs(self):
         # no metrics implemented for this dataset
        cwd = os.getcwd()
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg.SOLVER.BASE_LR = 0.00002
        self.cfg.SOLVER.MAX_ITER = (
            2000
        )  # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            256
        )  # faster,  dataset
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # 3 classes (data, fig, hazelnut)
        self.cfg.OUTPUT_DIR =  os.path.join(cwd,'segmentation_sessions',self.session_name)
        
    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
                    

# pollen_metadata = MetadataCatalog.get("pollen_dataset")
obj = RCNN_train(2,4,'test')
obj.train()




