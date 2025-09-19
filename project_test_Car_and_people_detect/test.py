from roboflow import Roboflow 
import ultralytics
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

import pandas as pd
import numpy as np


import cv2
#from IPython.display import display, Image, Video, HTML
import torch
from pytube import YouTube

import os
import subprocess

import random
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# from IPython import display
# display.clear_output()

from tqdm import tqdm

device =  'cuda:0' if torch.cuda.is_available() else 'cpu'

class Config():
    #crosswalk_folder = '/kaggle//working/crosswalk-2'
    #train_results = '/kaggle/working/runs/detect/train'
    #infe_video = '/kaggle/working/predestrian_walk.mp4'
    #output_path = './output_video.mp4'
    #output_path_sahi = './output_video_sahi.mp4'
    conf = 0.2
    weights = 'yolov8x.pt'


coco_classes = [0,2]#person and car
model = YOLO(Config.weights)
#weights_path = f'/kaggle/working/{Config.weights}' 

# sahi_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     #model_path=weights_path,
#     confidence_threshold=Config.conf,
#     device=device  
# )

# croswalk_points = {
    
#     1: np.array([[223,607], [135,400],[227,299], [570,147], [682,133], [826,148],[900,205], [428,603]],dtype =int),
#     2: np.array([[179,274], [37,199], [176,121],[491,136]],dtype =int),
#     3: np.array([[852,123],[725,108], [822,58], [900,81]],dtype =int),
    
# }

croswalk_points = {
    
    1: np.array([[242,330], [683,330], [683,428], [242,428]],dtype =int),
    
}







def build_sahi_results(result_sahi_obj):
    
    object_prediction_list  = result_sahi_obj.object_prediction_list

    res_sahi_list = []#bbox + class

    for ind, _ in enumerate(object_prediction_list):
        boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
            object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
        clss = object_prediction_list[ind].category.id
        conf = object_prediction_list[ind].score.value
        boxes = list(boxes)

        if clss in coco_classes:
            res_sahi_list.append(boxes + [conf,clss])
    
    return res_sahi_list


def createPolygon(img,dict_vertices):
    '''
    Draw a polygon in a image
    '''
    
    for cw_id,vertices in zip(dict_vertices.keys(),dict_vertices.values()):
        cv2.polylines(img, [vertices.reshape(-1,1,2)], True, (0,0,255), 0)

        mod = img.copy()
        mod = cv2.fillPoly(mod, pts = [vertices], color=(0,0,255))
        background = img.copy()
        overlay = mod.copy()

        img = cv2.addWeighted(background, 0.9,
                                overlay, 0.1,
                                0.1, overlay)

    
    return img



def detectPersonRisck(frame_df, dict_vertices, img,thickness=1):
        
        '''
        - Draw bbox for detections
        - Flag detections in Polygons Risks
        '''
        
        count_person_roi = 0
        risk_detections = [0 for x in range(len(frame_df))]      
        img = createPolygon(img=img,dict_vertices=dict_vertices)
        classes = frame_df['class'].values
        
        for i, person_detected in enumerate(frame_df[['xmin','xmax','ymin','ymax']].values.astype(int)):
            
            start_point = (person_detected[0], person_detected[-1]) # x_min, y_max
            end_point = (person_detected[1], person_detected[2]) # x_max, y_min
            class_detected = classes[i]
            title_x = int(person_detected[0] + ((person_detected[1]-person_detected[0])/2))
            title_y = int(person_detected[2] - (person_detected[2] *0.05))
            #print(start_point, end_point, class_detected,img.shape)

            
            if class_detected == 'car': 
                cv2.rectangle(img, start_point, end_point, (255,0,0), thickness) # red: in danger
                cv2.putText(img,'C', (title_x, title_y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 2) 
            if class_detected == 'person': 
                
                #print('person_detected',person_detected)   
                x_min, x_max = person_detected[0], person_detected[1]
                y_max = person_detected[-1]
                #print('x_min,y_max',x_min,y_max)

                foot_1 = (int(x_min),int(y_max))
                foot_2 = (int(x_max),int(y_max))

                ### draw little circle on each foot, for every person_detected
#                 cv2.circle(img,np.array(foot_1).astype(int) , 3, (255, 0, 0), -1)
#                 cv2.circle(img,np.array(foot_2).astype(int)  , 3, (255, 0, 0), -1) 

                ### Check if the point (foot) is inside the polygons
                person_detected_in_risk = False
                for cw_id,vertices in zip(dict_vertices.keys(),dict_vertices.values()):
                    inside1 = cv2.pointPolygonTest(vertices, foot_1, False) 
                    inside2 = cv2.pointPolygonTest(vertices, foot_2, False) 

                    
                    ### 1: inside ; 0: on border ; -1: outside
                    if inside1 == 1 or inside2 == 1:
                        person_detected_in_risk = True
                        count_person_roi += 1
                        risk_detections[i] = 1
                   
                
                
                color_bbox_person = (0,0,255) if person_detected_in_risk else (0,255,0)
                cv2.rectangle(img, start_point, end_point, color_bbox_person, thickness)
                cv2.putText(img,'P', (title_x, title_y), cv2.FONT_HERSHEY_TRIPLEX, 1, color_bbox_person, 2)
            if class_detected == 'crosswalk':
                cv2.rectangle(img, start_point, end_point, (28,226,233), thickness)
                
    
            
        return count_person_roi, risk_detections, img 




def pipeline_from_predictions(result_array, img):
    
    position_frame = pd.DataFrame(result_array, 
                               columns = ['xmin', 'ymin', 'xmax', 
                                          'ymax', 'conf', 'class'])

    position_frame['class'] = position_frame['class'].replace({0:'person', 2:'car'})
    
    count_person_roi, risk_detections,bbox_image = detectPersonRisck(position_frame, 
                                                                     croswalk_points, 
                                                                     img)

    video_height,video_width,_ = bbox_image.shape
    
    cv2.putText(bbox_image,f'Danger: {count_person_roi}', 
            (video_width-200,video_height-(video_height -100)), 
            cv2.FONT_HERSHEY_PLAIN, 
            2, 
            (0,0,255), 
            2)
    
    return bbox_image


def run():
    cap = cv2.VideoCapture('/home/ab123/opencvZoo2/project_test/data/거동이불편01.mp4')

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count)):
        frame_exists, frame = cap.read()

        results = model.predict(
            frame, 
            conf=Config.conf, 
            classes=coco_classes, 
            device=device, 
            verbose=False
        )

        bbox_image = pipeline_from_predictions(
            result_array=results[0].cpu().numpy().boxes.data,
            img=frame
        )
        
        cv2.imshow('YOLO output', bbox_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n영상 중단됨 (사용자 입력)\n")
            break
    cap.release()
    cv2.destroyAllWindows()




def main():
    run()

if __name__ == "__main__":
    main()




