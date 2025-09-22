import cv2
from detect_traffic import Detect_Traffic
from detect_crosswalk import Detect_Crosswalk
from detect_vehicle import Detect_Vehicle
from importpath import ImportPath

class MainVideo:
    def __init__(self):
        self.traffic_detector = Detect_Traffic(ImportPath.TRAFFIC_MODEL_PATH)
        self.crosswalk_detector = Detect_Crosswalk(ImportPath.CROSSWALK_MODEL_PATH)
        self.vehicle_detector = Detect_Vehicle(ImportPath.VEHICLE_MODEL_PATH)
    
    def videos_run(self):
        cap = cv2.VideoCapture('/home/inteee/opencvZoo2/imageproject/data/videos/full3.mp4')
    
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            #신호등 감지기
            frame1 = self.traffic_detector.detect_traffic(frame)
            frame2, crosswalk_point = self.crosswalk_detector.detect_crosswalk(frame1)
            print(crosswalk_point)
            frame3 = self.vehicle_detector.detect_vehicle(frame2, crosswalk_point)
            cv2.imshow("Result", frame3)
            if cv2.waitKey(1) == ord('q'):
                break

def main():
    app = MainVideo()
    app.videos_run()

if __name__ == "__main__":
    main()
    