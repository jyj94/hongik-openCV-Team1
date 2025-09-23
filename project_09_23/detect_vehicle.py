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
        self.traffic_status = "신호등 탐지 안됨"
        self.warning = None
        
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
        """폴리곤 그리기 (list of np.array 또는 단일 np.array)"""
        # None이나 빈 배열이면 바로 리턴
        if vertices_list is None:
            return img
        if isinstance(vertices_list, np.ndarray):
            if vertices_list.size == 0:
                return img
            vertices_list = [vertices_list]  # list로 감싸기

        for vertices in vertices_list:
            if vertices is None or len(vertices) == 0:
                continue
            pts = vertices.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 0, 255), 1)

            # 반투명 채우기
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
            img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)

        return img
    

    def _detect_person_risk(self, frame_df, vertices_list, img, thickness=2):
        """사람, 차량 bbox를 그리고 위험영역 안/밖을 판별"""
        count_person_roi = 0
        risk_detections = [0 for _ in range(len(frame_df))]

        # polygon 먼저 그림 (없으면 그냥 넘어감)
        img = self._create_polygon(img, vertices_list)

        classes = frame_df["class"].values

        for i, det in enumerate(frame_df[["xmin", "xmax", "ymin", "ymax"]].values.astype(int)):
            x_min, x_max, y_min, y_max = det
            start_point = (x_min, y_max)  # bottom-left
            end_point = (x_max, y_min)    # top-right
            class_detected = classes[i]

            title_x = int(x_min + (x_max - x_min) / 2)
            title_y = int(y_min - 10)
            self.warning = None

            # 차량의 경우
            if class_detected in ["car", "truck", "bus"]:
                in_crosswalk = False
                for pts in vertices_list:
                    if pts is None or len(pts) == 0:
                        continue
                    pts = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                    # 차량 bbox 중심 하단 점
                    car_bottom = (float((x_min + x_max) / 2), float(y_max))
                    inside = cv2.pointPolygonTest(pts, car_bottom, False)
                    if inside >= 0:
                        in_crosswalk = True
                        break

                if self.traffic_status == "빨간불":
                    color = (255, 0, 0)  # 파랑
                    
                elif self.traffic_status == "초록불":
                    if in_crosswalk:
                        color = (0, 0, 255)  # 빨강
                        # 위험신호 재생 코드
                        self.warning = "차량"
                        
                    else:
                        color = (255, 0, 0)  # 파랑
                else:  # 신호등 탐지 안됨
                    color = (255, 0, 0)  # 파랑

                cv2.rectangle(img, start_point, end_point, color, thickness)
                cv2.putText(img, "Car", (title_x, title_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                

            
            # 사람인 경우
            elif class_detected == "person":
                foot_1 = (float(x_min), float(y_max))
                foot_2 = (float(x_max), float(y_max))

                in_crosswalk = False
                for pts in vertices_list:
                    if pts is None or len(pts) == 0:
                        continue
                    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    inside1 = cv2.pointPolygonTest(pts, foot_1, False)
                    inside2 = cv2.pointPolygonTest(pts, foot_2, False)
                    if inside1 >= 0 or inside2 >= 0:
                        in_crosswalk = True
                        break

                # 신호 상태별 처리
                if self.traffic_status == "빨간불":
                    if in_crosswalk:
                        color = (0, 0, 255)  # 빨강
                        count_person_roi += 1
                        risk_detections[i] = 1
                        # 위험신호 재생코드
                        self.warning = "사람"
                    else:
                        color = (0, 255, 0)  # 초록
                elif self.traffic_status == "초록불":
                    color = (0, 255, 0)  # 초록 (항상)
                else:  # 신호등 탐지 안됨
                    color = (0, 255, 0)  # 초록

                cv2.rectangle(img, start_point, end_point, color, thickness)
                cv2.putText(img, "Person", (title_x, title_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


            # if class_detected == "car":
            #     cv2.rectangle(img, start_point, end_point, (255, 0, 0), thickness)
            #     cv2.putText(img, "Car", (title_x, title_y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # elif class_detected == "person":
            #     foot_1 = (int(x_min), int(y_max))
            #     foot_2 = (int(x_max), int(y_max))

            #     in_risk = False
                
            #     for pts in vertices_list:
            #         if pts is None or len(pts) == 0:
            #             continue
            #         pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

            #         # pointPolygonTest는 직사각형도 그대로 동작
            #         inside1 = cv2.pointPolygonTest(pts, foot_1, False)
            #         inside2 = cv2.pointPolygonTest(pts, foot_2, False)

            #         # if self.traffic_status == "신호등 탐지 안됨":
            #         # # 신호등 정보 없으면 모두 그리기    




            #         if inside1 >= 0 or inside2 >= 0:  # 내부 또는 경계선
            #             in_risk = True
            #             count_person_roi += 1
            #             risk_detections[i] = 1

            #     color = (0, 0, 255) if in_risk else (0, 255, 0)
            #     cv2.rectangle(img, start_point, end_point, color, thickness)
            #     cv2.putText(img, "Person", (title_x, title_y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return count_person_roi, risk_detections, img

    def _pipeline_from_predictions(self, result_array, img):
        """YOLO 결과를 DataFrame으로 변환 후 risk 계산"""
        if result_array is None or len(result_array) == 0:
            return img  # 탐지 결과 없으면 그대로 리턴

        # 클래스 이름 가져오기
        class_names = [self.model.names[int(cls_id)] for cls_id in result_array[:, 5]]

        # DataFrame 생성
        df = pd.DataFrame(result_array,
                        columns=["xmin", "ymin", "xmax", "ymax", "conf", "class_id"])
        df["class"] = class_names  # class id 대신 이름 사용

        # 위험 영역 판단
        count_person_roi, _, bbox_image = self._detect_person_risk(df, self.croswalk_points, img)

        # 위험 인원 수 표시
        h, w, _ = bbox_image.shape
        cv2.putText(bbox_image, f"Danger: {count_person_roi}",
                    (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

        return bbox_image

    def detect_vehicle(self, frame, croswalk_points, traffic_status):
        """main.py에서 호출하는 공통 인터페이스"""
        self.traffic_status = "신호등 탐지 안됨" # 신호등 값 초기화

        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=self.coco_classes,
            device=device,
            verbose=False
        )

        self.croswalk_points = croswalk_points
        self.traffic_status = traffic_status

        frame_out = self._pipeline_from_predictions(
            result_array=results[0].cpu().numpy().boxes.data,
            img=frame.copy()
        )
        return frame_out, self.warning
