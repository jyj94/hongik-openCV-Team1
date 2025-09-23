import cv2
from detect_traffic import Detect_Traffic
from detect_crosswalk import Detect_Crosswalk
from detect_vehicle import Detect_Vehicle
from importpath import ImportPath
import numpy as np

class MainVideo:
    def __init__(self):
        self.traffic_detector = Detect_Traffic(ImportPath.TRAFFIC_MODEL_PATH)
        self.crosswalk_detector = Detect_Crosswalk(ImportPath.CROSSWALK_MODEL_PATH)
        self.vehicle_detector = Detect_Vehicle(ImportPath.VEHICLE_MODEL_PATH)
        self.traffic_status = "신호등 탐지 안됨"

    def videos_run(self):
        cap = cv2.VideoCapture('/home/ab123/opencvZoo2/새 폴더/Case1-1.mp4')
    
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            #신호등 및 1차 횡단보도 탐지기
            frame1, crosswalk_point1, self.traffic_status = self.traffic_detector.detect_traffic(frame)
            print(self.traffic_status)
            
            frame2, crosswalk_point2 = self.crosswalk_detector.detect_crosswalk(frame1)
            #print(crosswalk_point)

            if len(crosswalk_point1) > 0:
                frame3 = self.vehicle_detector.detect_vehicle(frame2, crosswalk_point1, self.traffic_status)
            elif len(crosswalk_point2) > 0:
                frame3 = self.vehicle_detector.detect_vehicle(frame2, crosswalk_point2, self.traffic_status)
            else:
                frame3 = frame2


 


            
            cv2.imshow("Result", frame3)
            if cv2.waitKey(1) == ord('q'):
                break

def main():
    app = MainVideo()
    app.videos_run()

if __name__ == "__main__":
    main()
    