import os


# import some common libraries
import numpy as np
import cv2
import time
import pyclipper
import uuid
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from imantics import Polygons, Mask

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Predictor:

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.num_classes = 4
        self.define_model()
        self.nms_thresh = 0.5
        pass

    def define_model(self):
        self.cfg = get_cfg()
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.DATALOADER.NUM_WORKERS = 2
        # path to the model we just trained
        self.cfg.MODEL.WEIGHTS = self.weights_path
        # set a custom testing thresholds
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        # self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST =0.01
        self.prepare_configs()
        self.predictor = DefaultPredictor(self.cfg)

    def prepare_configs(self):
        # no metrics implemented for this dataset

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            512
        )  # faster,  dataset
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes

    def nms(self, boxes, confs, min_mode=False):

        boxes = np.array(boxes)
        confs = np.array(confs)
        order = confs.argsort()[::-1]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        keep = []

        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def pad_image(self, image, size=(256, 256)):
        h, w, c = image.shape
        pad_l = pad_r = int((size[1]-w)/2)
        pad_t = pad_b = int((size[0]-h)/2)
        img = cv2.copyMakeBorder(image.copy(), pad_t, pad_b,
                                 pad_l, pad_r, cv2.BORDER_CONSTANT)
        img = cv2.resize(img, size)
        return img

    def remove_padding(self, img, tol=0):
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img > tol
        if img.ndim == 3:
            mask = mask.all(2)
        m, n = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
        row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
        return img[row_start:row_end, col_start:col_end]

    def predict(self, img_path, plot=False):

        im = cv2.imread(img_path)
        outputs = self.predictor(im)

        pred_classes = outputs['instances'].pred_classes.to("cpu").numpy()
        pred_boxes = outputs['instances'].pred_boxes.to("cpu")
        scores = outputs['instances'].scores.to("cpu").numpy()
        mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        output_boxes = []

        # pol_points=[]
        # output_boxes = []
        # for i in mask_array:
        #     polygons = Mask(i).polygons()
        #     pol_points.append(polygons.points)

        for box, label in zip(pred_boxes, pred_classes):
            box = box.cpu().detach().numpy()

            # label = map_label[str(label)]
            x1, y1, x2, y2 = box.astype('int')
            output_boxes.append([label, x1, y1, x2, y2])

        keep = self.nms(output_boxes, scores)
        if len(keep) > 0:
            output_boxes = np.array(output_boxes)[keep]
            mask_array = np.array(mask_array)[keep]
            pred_classes = np.array(pred_classes)[keep]

        # Remove Boxes inside
        keep2 = self.check_inner(output_boxes)
        if len(keep2) > 0:
            output_boxes = np.array(output_boxes)[keep2]
            mask_array = np.array(mask_array)[keep2]
            pred_classes = np.array(pred_classes)[keep2]

        map_label = {'3': 'POL',
                     '1': 'ANOM',
                     '0': 'OTHPAR',
                     '2': 'MOL'}

        pol_points = []

        for i in mask_array:
            polygons = Mask(i).polygons()
            pol_points.append(polygons.points)

        pol_points_offset = []
        for pol_point in pol_points:
            pol_tup = tuple(tuple(sub) for sub in pol_point[0].tolist())
            # print(pol_tup)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(pol_tup, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            contours_list = pco.Execute(2.0)
            
            pol_points_offset.append(np.array(contours_list[0]))
            # print(contours_list)

        self.pol_plot(im, pol_points_offset, output_boxes, img_path)
        # self.pol_plot(im, pol_points, output_boxes, img_path)
        exit()
        for mask, pred_class in zip(mask_array, pred_classes):

            label = map_label[str(pred_class)]
            if label == 'POL' or label == 'MOL':
                # mask = mask
                masked = cv2.bitwise_and(im, im, mask=mask.astype("uint8"))
                tmp_ = self.remove_padding(masked)
                output = self.pad_image(tmp_)

                if label == 'POL':
                    udi = str(uuid.uuid4())+'.png'
                    output_path = os.path.join('masked', 'POL', udi)
                if label == 'MOL':
                    udi = str(uuid.uuid4())+'.png'
                    output_path = os.path.join('masked', 'MOL', udi)
                print(output_path)
                cv2.imwrite(output_path, output)
                # if cv2.waitKey(0) & 0xFF == ord('q'):

        # if plot:
        #     self.pol_plot(im,pol_points,output_boxes,img_path)
        # return pol_points,output_boxes
        return None

    def check_inner(self, boxes):
        if type(boxes) is np.ndarray:
            boxes = boxes.tolist()
        to_remove = []
        to_remove_ind = []
        for i in boxes:
            for j in boxes:
                if i != j:
                    x1, y1, x2, y2 = i[1:]
                    a_small = (x2 - x1) * (y2 - y1)
                    point = Point((x1+x2)/2, (y1+y2)/2)

                    x1, y1, x2, y2 = j[1:]
                    a_large = (x2 - x1) * (y2 - y1)
                    poly_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    polygon = Polygon(poly_points)

                    if polygon.contains(point) and a_large > a_small:
                        to_remove.append(i)
        kpt = list(range(len(boxes)))

        for t_id in to_remove:
            ind = boxes.index(t_id)
            try:
                kpt.remove(ind)
            except:
                pass
            to_remove_ind.append(ind)

        return kpt

    def pol_plot(self, im, pol_points, boxes, img_path):

        color = (255, 0, 0)
        thickness = 1
        isClosed = False

        map_label = {
                    '3': 'POL',
                    '1': 'ANOM',
                    '0': 'OTHPAR',
                    '2': 'MOL'}
        map_color = {
                'POL': (255, 0, 0),  # BLUE
                'ANOM': (0, 0, 255),  # RED
                'OTHPAR': (0, 255, 0),  # GREEN
                'MOL': (0, 255, 255)}  # Yellow

        for points, box in zip(pol_points, boxes):
            label, x1, y1, x2, y2 = box
            points = np.array([points])

            label = map_label[str(label)]
            color = map_color[str(label)]
            im = cv2.polylines(im, points, isClosed, color, thickness)
            cv2.putText(im, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .3, 255)
            _ = cv2.rectangle(im, (x1, y1), (x2, y2),
                                  (255, 255, 255), thickness=1)

       
        return 0

        # img_name = img_path.split('/')[-1]
        # os.makedirs('out_predict', exist_ok=True)
        # cv2.imwrite('out_predict/'+img_name, im)


if __name__ == '__main__':
    weights = '/home/laanta/sagun/segmentation/output/model_jan3.pth'
    folder_path = '/home/laanta/sagun/yolo/inference/test'
    obj = Predictor(weights)

    for img in sorted(os.listdir(folder_path)):
        if '.png' in img:
            # img = '/home/laanta/sagun/yolo/inference/test/20200214T061825969-1901-0x2.png'

            img_path = os.path.join(folder_path, img)
            print(img_path)
            out = obj.predict(img_path, plot=True)
            # exit()

