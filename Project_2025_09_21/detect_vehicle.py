import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 위험영역(횡단보도 좌표 예시)
# croswalk_points = {
#     1: np.array([[242, 330], [683, 330], [683, 428], [242, 428]], dtype=int),
# }


class Detect_Vehicle:
    def __init__(self, model_path, font_path=None, conf=0.2):
        self.model = YOLO(model_path)
        self.conf = conf
        self.coco_classes = [0, 2]  # 0: person, 2: car
        self.font_path = font_path
        self.croswalk_points = []

    # def _create_polygon(self, img, dict_vertices):
    #     """횡단보도 영역을 그림"""
    #     for vertices in dict_vertices.values():
    #         cv2.polylines(img, [vertices.reshape(-1, 1, 2)], True, (0, 0, 255), 1)
    #         mod = img.copy()
    #         mod = cv2.fillPoly(mod, pts=[vertices], color=(0, 0, 255))
    #         background = img.copy()
    #         overlay = mod.copy()
    #         img = cv2.addWeighted(background, 0.9, overlay, 0.1, 0.1, overlay)
    #     return img
    
    def _create_polygon(self, img, vertices_list):
        """
        img : 이미지
        vertices_list : np.array([[x1,y1],[x2,y2],...])
        """
        if not vertices_list:  # 비어있으면 바로 리턴
            return img
        
         # 리스트라면 numpy array로 변환
        if isinstance(vertices_list, list):
            vertices_list = np.array(vertices_list, dtype=np.int32)
        
        # 선 그리기
        cv2.polylines(img, [vertices_list.reshape(-1,1,2)], True, (0,0,255), 1)

        # 투명 채우기
        overlay = img.copy()
        cv2.fillPoly(overlay, [vertices_list], color=(0,0,255))
        img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)

        return img
    

    def _detect_person_risk(self, frame_df, vertices_list, img, thickness=2):
        """사람, 차량 bbox를 그리고 위험영역에 있는지 판단"""
        count_person_roi = 0
        risk_detections = [0 for _ in range(len(frame_df))]
        # polygon 그리기
        img = self._create_polygon(img, vertices_list)
        classes = frame_df["class"].values

        for i, det in enumerate(frame_df[["xmin", "xmax", "ymin", "ymax"]].values.astype(int)):
            x_min, x_max, y_min, y_max = det
            start_point = (x_min, y_max)  # bottom-left
            end_point = (x_max, y_min)    # top-right
            class_detected = classes[i]

            title_x = int(x_min + (x_max - x_min) / 2)
            title_y = int(y_min - 10)

            if class_detected == "car":
                cv2.rectangle(img, start_point, end_point, (255, 0, 0), thickness)
                cv2.putText(img, "Car", (title_x, title_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            elif class_detected == "person":
                foot_1 = (int(x_min), int(y_max))
                foot_2 = (int(x_max), int(y_max))

               # 횡단보도 안에 있는지 체크
                in_safe = False  # 폴리곤 안이면 True
                for vertices in vertices_list:
                    pts = np.array(vertices, dtype=np.int32)
                    inside1 = cv2.pointPolygonTest(pts, foot_1, False)
                    inside2 = cv2.pointPolygonTest(pts, foot_2, False)
                    if inside1 == 1 or inside2 == 1:
                        in_safe = True  # 폴리곤 안이면 안전

                # 색상 결정: 폴리곤 안이면 초록(안전), 바깥이면 빨강(위험)
                color = (0, 255, 0) if in_safe else (0, 0, 255)
                cv2.rectangle(img, start_point, end_point, color, thickness)
                cv2.putText(img, "Person", (title_x, title_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 위험 카운트는 폴리곤 밖에 있을 때 증가
                if not in_safe:
                    count_person_roi += 1
                    risk_detections[i] = 1

        return count_person_roi, risk_detections, img

    def _pipeline_from_predictions(self, result_array, img):
        """YOLO 결과를 DataFrame으로 변환 후 risk 계산"""
        df = pd.DataFrame(result_array,
                          columns=["xmin", "ymin", "xmax", "ymax", "conf", "class"])
        df["class"] = df["class"].replace({0: "person", 2: "car"})

        count_person_roi, _, bbox_image = self._detect_person_risk(df, self.croswalk_points, img)

        h, w, _ = bbox_image.shape
        cv2.putText(bbox_image, f"Danger: {count_person_roi}",
                    (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        return bbox_image

    def detect_vehicle(self, frame, croswalk_points):
        """main.py에서 호출하는 공통 인터페이스"""
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=self.coco_classes,
            device=device,
            verbose=False
        )

        self.croswalk_points = croswalk_points
        
        frame_out = self._pipeline_from_predictions(
            result_array=results[0].cpu().numpy().boxes.data,
            img=frame.copy()
        )
        return frame_out