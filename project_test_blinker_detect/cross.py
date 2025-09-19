import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator


# 횡단보도,신호등 모델
MODEL_PATH = '/home/ab123/opencvZoo2/project_test/run/weights/best.pt'

img_size = 640
conf_thres = 0.009  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = None  # filter by class
agnostic_nms = False  # class-agnostic NMS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
class_names = ['횡단보도', '빨간불', '초록불'] # model.names
stride = int(model.stride.max())
colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0)) # (gray, red, green)

net = cv2.dnn.readNetFromDarknet('/home/ab123/opencvZoo2/project_test/models/yolov4-ANPR.cfg','/home/ab123/opencvZoo2/project_test/models/yolov4-ANPR.weights')

model2_path = '/home/ab123/opencvZoo2/project_test/data/Crosswalks_ONNX_Model.onnx'


crosswalk_point = []

# 동영상 로드

cap = cv2.VideoCapture('/home/ab123/opencvZoo2/project_test/data/거동이불편01.mp4')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('data/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    H, W, _ = img.shape

    # preprocess
    img_input = letterbox(img, img_size, stride=stride)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()
    img_input /= 255.
    img_input = img_input.unsqueeze(0)

    # inference 횡단보도,신호등
    pred = model(img_input, augment=False, visualize=False)[0]

    # postprocess
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    pred = pred.cpu().numpy()

    pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()

    boxes, confidences, class_ids = [], [], []

   
    print(pred)
    # Visualize
    annotator = Annotator(img.copy(), line_width=3, example=str(class_names), font='data/NanumPenScript-Regular.ttf')

    cw_x1, cw_x2 = None, None # 횡단보도 좌측(cw_x1), 우측(cw_x2) 좌표

    if len(pred) <= 0 :
        print("탐지 불가")

    for p in pred:
        class_name = class_names[int(p[5])]
        x1, y1, x2, y2 = p[:4]

        annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])


        if class_name == '초록불':
            print("초록불 감지됨")

        elif class_name == '빨간불':
            print("빨간불 건너지 마세요")

        elif class_name == '횡단보도':
            cw_x1, cw_x2 = x1, x2
            # 횡단보도 bbox로 4개의 점 가져오기
            cw_X1, cw_Y1, cw_X2, cw_Y2 = cw_x1, y1, cw_x2, y2
            crosswalk_point = np.array([
                        [cw_x1, cw_Y1],  # 좌상단
                        [cw_x2, cw_Y1],  # 우상단
                        [cw_x2, cw_Y2],  # 우하단
                        [cw_x1, cw_Y2]   # 좌하단
                    ], dtype=int)
            print(f"x1 : {cw_X1}, x2 : {cw_X2}, y1 : {cw_Y1}, y2 : {cw_Y2}")




    result_img = annotator.result()

    cv2.imshow('result', result_img)
    out.write(result_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()