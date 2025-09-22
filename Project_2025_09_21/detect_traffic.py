import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from importpath import ImportPath
import sys
sys.path.append(ImportPath.TRAFFIC_MODEL_PATH)


class Detect_Traffic:
    def __init__(self, Model_path, img_size=640, conf_thres=0.5, iou_thres=0.45):
        self.img_size = img_size
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class
        self.agnostic_nms = False  # class-agnostic NMS

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ckpt = torch.load(Model_path, map_location=self.device, weights_only=False)
        self.model = self.ckpt['ema' if self.ckpt.get('ema') else 'model'].float().fuse().eval()
        self.class_names = ['횡단보도', '빨간불', '초록불'] # model.names
        self.stride = int(self.model.stride.max())
        self.colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0)) # (gray, red, green)
         
    def preprocess_blinker(self, frame):
        # preprocess
        img_input = letterbox(frame, self.img_size, stride=self.stride)[0]
        img_input = img_input.transpose((2, 0, 1))[::-1]
        img_input = np.ascontiguousarray(img_input)
        img_input = torch.from_numpy(img_input).to(self.device)
        img_input = img_input.float()
        img_input /= 255.
        img_input = img_input.unsqueeze(0)
        return img_input

    def infer(self, frame):

        # inference 횡단보도,신호등
        img_input = self.preprocess_blinker(frame)
        
        pred = self.model(img_input, augment=False, visualize=False)[0]

        # postprocess
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)[0]

        pred = pred.cpu().numpy()
 
        pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], frame.shape).round()
        
        return pred

    def detect_traffic(self, frame):
        
        pred = self.infer(frame)
        
         # Visualize
        annotator = Annotator(frame.copy(), line_width=3, example=str(self.class_names), font=ImportPath.FONT_PATH)

        cw_x1, cw_x2 = None, None # 횡단보도 좌측(cw_x1), 우측(cw_x2) 좌표

        if pred is not None:
            for p in pred:
                cls_id = int(p[5])
                class_name = self.class_names[cls_id]
                x1, y1, x2, y2 = p[:4]
                annotator.box_label([x1,y1,x2,y2], f"{class_name} {p[4]*100:.1f}%", color=self.colors[cls_id])
              
        result_img = annotator.result()
                
        return result_img
        
    