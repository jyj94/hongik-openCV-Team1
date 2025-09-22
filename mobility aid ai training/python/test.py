import cv2
from ultralytics import YOLO


model = YOLO('../mobility aid ai training/python/mobility_aid_best.pt').to('cuda')
#model2 = YOLO('/home/kjonghun0828/mobility training/python/yolov4-ANPR.weights').to('cuda')
video_file_path = "../mobility aid ai training/test.mp4"
#test.mp4는 google drive cctv에서 다운로드. 
# results = model(video_file_path, save = True)
cap = cv2.VideoCapture(video_file_path)

while True: 
    ret, frame = cap.read() 
    if not ret: 
        break 
    results = model(frame) 
    annotated_frame = results[0].plot() 
    cv2.imshow('YOLO Object Detection', annotated_frame) 
        
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

cap.release() 
cv2.destroyAllWindows()
print(results)






