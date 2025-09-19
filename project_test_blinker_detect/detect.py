import ultralytics
from ultralytics import YOLO


class detect():

    class config():
        conf = 0.2
        weights = 'yolo8x.pt'



    def __init__(self, device, croswalk_points):
        self.config()
        self.device = device
        self.croswalk_points = croswalk_points
        self.coco_classes = [0, 2] #person and car
        self.model = YOLO(self.config.weight)

    def build_sahi_results(self, result_sahi_obj):
    
        object_prediction_list  = result_sahi_obj.object_prediction_list

        res_sahi_list = []#bbox + class

        for ind, _ in enumerate(object_prediction_list):
            boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
                object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
            clss = object_prediction_list[ind].category.id
            conf = object_prediction_list[ind].score.value
            boxes = list(boxes)

            if clss in self.coco_classes:
                res_sahi_list.append(boxes + [conf,clss])
        
        return res_sahi_list