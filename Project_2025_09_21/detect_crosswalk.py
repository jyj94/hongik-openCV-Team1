# pt 파일 경로 설정
# 장치 번호 바꾸기
import cv2
import numpy as np
from ultralytics import YOLO


class Detect_Crosswalk:
    def __init__(self, model_path):
        """
        횡단보도 감지 전용 YOLO 모델 초기화
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.box_color = (0 , 255, 0) # 초록색
        
    def detect_crosswalk(self, frame):
        """
        한 프레임에서 횡단보도 감지
        반환값: 
            frame_out: 시각화된 프레임
            crosswalk_points: 횡단보도 영역 좌표 리스트 (np.array)
        """
        frame = frame.copy()
        results = self.model(frame, verbose=False)
        
        crosswalk_points = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.class_names[int(cls)]} {conf:.2f}"

                # 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, self.box_color, 2)

                # 횡단보도 영역 좌표 계산 (좌상단, 우상단, 우하단, 좌하단)
                points = np.array([
                    [x1, y1],  # 좌상단
                    [x2, y1],  # 우상단
                    [x2, y2],  # 우하단
                    [x1, y2],  # 좌하단
                ], dtype=int)
 
                crosswalk_points.append(points)

        return frame, crosswalk_points
        