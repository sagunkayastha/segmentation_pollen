import os
import json
import cv2
import numpy as np

from predict import Predictor
from utils.coco2lbl import CocoDatasetHandler

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class create_json:
    def __init__(self, image_folder, weights, output_filename, label_me_output):
        self.output = {}
        self.image_folder = image_folder
        self.model = Predictor(weights)
        self.output_filename = output_filename
        self.label_me_output = label_me_output

    def add_categories(self):
        categories = [
            {
                "supercategory": "ANOM",
                "id": 2,
                "name": "ANOM"
            },
            {
                "supercategory": "OTHPAR",
                "id": 1,
                "name": "OTHPAR"
            },
            {
                "supercategory": "MOL",
                "id": 3,
                "name": "MOL"
            },
            {
                "supercategory": "POL",
                "id": 4,
                "name": "POL"
            }
        ]
        return categories

    def add_images(self):
        id_ = 1
        tmp_list = []
        cwd = os.getcwd()
        for file_ in os.listdir(self.image_folder):
            new_dict = {}
            img_path = os.path.join(cwd, self.image_folder, file_)
            im = cv2.imread(img_path)
            height, width = im.shape[0], im.shape[1]
            new_dict['height'] = height
            new_dict['width'] = width
            new_dict['id'] = id_
            new_dict['file_name'] = img_path   # imagepath
            id_ += 1
            tmp_list.append(new_dict)

        return tmp_list

    def add_annotations(self):
        ann_id = 0
        for images in self.output['images']:
            img_path = images['file_name']
            img_id = images['id']
            print(img_path, img_id)
            segpoints, bboxpoints = self.model.predict(img_path, plot=False)

            for i, j in zip(segpoints, bboxpoints):
                newdict2 = {}
                newdict2['iscrowd'] = 0
                newdict2['image_id'] = int(img_id)

                x, y, max_x, max_y = j[1:]
                width = max_x - x
                height = max_y - y
                newdict2['bbox'] = [x, y, width, height]

                final2 = i[0].reshape(-1).tolist()

                newdict2['segmentation'] = [final2]
                newdict2['category_id'] = int(j[0])+1
                ann_id += 1
                newdict2['id'] = ann_id

                newdict2['area'] = images['height']*images['width']
                self.output['annotations'].append(newdict2)

    def run(self):
        self.output['images'] = self.add_images()
        self.output['categories'] = self.add_categories()
        self.output['annotations'] = []
        self.add_annotations()
        self.save_json()
        self.convert_to_lableme()

    def save_json(self):
        os.makedirs('Outputs/', exist_ok=True)
        self.coco_file_path = os.path.join('Outputs', self.output_filename)
        with open(self.coco_file_path, 'w') as fp:
            json.dump(self.output, fp, cls=NpEncoder)

    def convert_to_lableme(self):
        self.c2l = CocoDatasetHandler(self.coco_file_path, self.image_folder)
        self.c2l.coco2labelme()
        self.c2l.save_labelme(self.c2l.labelme.keys(), self.label_me_output)


image_folder = 'test'
model_path = '/home/laanta/sagun/segmentation/output/model_jan2.pth'
output_filename = 'coco.json'
label_me_output = 'Outputs/det_to_labelme'
obj = create_json(image_folder, model_path, output_filename, label_me_output)
output = obj.run()

