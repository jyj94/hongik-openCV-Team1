import sys, requests, numpy as np, cv2
from PySide6.QtWidgets import QApplication, QLCDNumber, QPushButton, QLabel, QFileDialog, QMessageBox, QTextEdit
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import QFile, Qt, QTimer, QUrl
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QPixmap
from detect_traffic import Detect_Traffic
from detect_crosswalk import Detect_Crosswalk
from detect_vehicle import Detect_Vehicle
from importpath import ImportPath
from PySide6.QtMultimedia import QSoundEffect
from datetime import datetime



class CrosswalkSign():
    def __init__(self, signWidget, LCDWidget, startCount=10, blinkInterval=500):
        self.sign = signWidget
        self.LCD = LCDWidget
        
        self.greenLightOnImg = QPixmap('/home/ab123/opencvZoo2/project_09_22/data/GUI_image/greenLightOn.png')
        self.greenLightOffImg = QPixmap('/home/ab123/opencvZoo2/project_09_22/data/GUI_image/greenLightOff.png')
        self.redLightImg = QPixmap('/home/ab123/opencvZoo2/project_09_22/data/GUI_image/redLight.png')

        self.isGreen = True  # 신호등이 초록 불인지
        self.greenOn = True  # 초록불 깜빡임 상태
        self.countNum = startCount
        self.startCount = startCount
        self.LCDVisible = True  # LCD 깜빡임 상태

        # LCD 초기값
        #self.setLCDValue(self.countNum)

        # 타이머 설정
        self.blinkTimer = QTimer()
        self.blinkTimer.timeout.connect(self.blink)
        self.blinkTimer.start(blinkInterval)  # ms
        

    def setLCDValue(self, val):
        self.LCD.display(val)

    def blink(self):
        # 신호등 초록불 깜빡이
        if self.isGreen:
            if self.greenOn and self.countNum < self.startCount // 2:
                self.sign.setPixmap(self.greenLightOffImg)
            else:
                self.sign.setPixmap(self.greenLightOnImg)
            self.greenOn = not self.greenOn
        else:
            self.sign.setPixmap(self.redLightImg)

        # LCD 깜빡이 & 감소
        if self.LCDVisible:
            self.LCD.setStyleSheet("color: #072908;")
            
        else:
            self.countNum -= 1
            if self.countNum <= 0:
                self.sign.setPixmap(self.redLightImg)
                self.setLCDValue('')
                self.blinkTimer.stop()
                return
            self.setLCDValue(self.countNum)
            if self.countNum < self.startCount // 2:
                self.LCD.setStyleSheet("color: #20b220;")
        self.LCDVisible = not self.LCDVisible
            
class MainWindow():
    def __init__(self):
        #변수 선언
        self.lightIsGreen = True
        self.currentFrame = -1
        self.MaximumFrame = 0
        self.crosswalk = None
        self.lastUpdate = datetime.now()
        
        #어플리케이션 객체 생성 및 ui 파일 불러오기
        self.app = QApplication([])
        uiFile = QFile('/home/ab123/opencvZoo2/project_09_22/GUI/mainWindow.ui')
        uiFile.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.window = loader.load(uiFile)
        uiFile.close()
        
        # 경고음 재생 클래스 초기화
        self.effect = QSoundEffect()
        self.effect.setSource(QUrl.fromLocalFile("/home/ab123/opencvZoo2/project_09_22/data/Sound/경고음.wav"))
        self.effect.setVolume(0.5)


        #GUI 위젯 생성 및 초기화

        self.textEdit = self.window.findChild(QTextEdit, "textEdit")
        # self.testButton = self.window.findChild(QPushButton, "testButton")
        # print(self.testButton)

        # self.testButton.clicked.connect(self.testButtonAct)
        
        
        self.videoFileButton = self.window.findChild(QPushButton, "videoFileButton")
        self.videoFileButton.clicked.connect(self.openVideoFile)
        
        self.graphicsView = self.window.findChild(QGraphicsView, "graphicsView")
        self.graphicsScene = QGraphicsScene()
        self.graphicsView.setScene(self.graphicsScene)
        self.pixmapItem = QGraphicsPixmapItem()
        self.graphicsScene.addItem(self.pixmapItem)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)
        
        # 탐지 객체들 초기화
        self.traffic_detector = Detect_Traffic(ImportPath.TRAFFIC_MODEL_PATH)
        self.crosswalk_detector = Detect_Crosswalk(ImportPath.CROSSWALK_MODEL_PATH)
        self.vehicle_detector = Detect_Vehicle(ImportPath.VEHICLE_MODEL_PATH)

        # 신호등 값
        self.traffic_status = "신호등 탐지 안됨"   

        # 경고음 재생 여부
        self.warning = False

    def setText(self, text):
        self.textEdit.append(str(datetime.now()) + " : " + text)

    def run(self):
        self.window.show()
        sys.exit(self.app.exec())
    
    def testButtonAct(self):
        self.crosswalk = CrosswalkSign(self.signLabel, self.countLCD, startCount=8, blinkInterval=500)
        
    def testCCTVLive(self):
        pass
        
            
    def openVideoFile(self):
        filePath,_ = QFileDialog.getOpenFileName(
            self.window,
            "Select a video file.",
            "./",
            "Videos (*.mp4 *.MP4 *.mkv *.MKV)"
        )
        if filePath:
            self.videoCap = cv2.VideoCapture(filePath)
            if not self.videoCap.isOpened():
                QMessageBox.warning(self.window, "Error", "Can't open video.")
                return
        else:
            return

        #비디오 변수 초기화
        #self.fps = self.videoCap.get(cv2.CAP_PROP_FPS)
        self.currentFrame = 0
        self.MaximumFrame = int(self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT))        
        #print(f'selected file: {filePath}')
        #print(f"해상도: {self.videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        #print(f"FPS: {self.videoCap.get(cv2.CAP_PROP_FPS)}")
        #print(f"총 프레임: {self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        self.textEdit.setPlainText('')
        
        self.nextFrame()
        self.graphicsScene.setSceneRect(self.pixmapItem.boundingRect())
        self.graphicsView.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        fps = 60
        interval = int(1000 / fps) if fps > 0 else 33
        self.timer.start(interval)
    
    def nextFrame(self, ret = None, frame = None):
        # 비디오가 열려 있는지 확인
        if not hasattr(self, 'videoCap') or not self.videoCap.isOpened():
            return
        
        # 한 프레임 읽기
        ret, frame = self.videoCap.read()
        if not ret:
            # 비디오 끝나면 처음으로 되돌리기
            self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.videoCap.read()
            self.currentFrame = 0
        else:
            self.currentFrame += 1
        #print(self.currentFrame)
        # -------------------------------
        # 프레임 처리 단계
        # -------------------------------
        # 1. 신호등 감지 (traffic_detector 필요)
        if hasattr(self, 'traffic_detector'):
            frame1, crosswalk_point1, self.traffic_status = self.traffic_detector.detect_traffic(frame)
            print(crosswalk_point1)
        else:
            frame1 = frame
            crosswalk_point1 = None

        # 2. 횡단보도 감지 (crosswalk_detector 필요)
        if hasattr(self, 'crosswalk_detector'):
            frame2, crosswalk_point2 = self.crosswalk_detector.detect_crosswalk(frame1)
            # 디버깅용
            #print(crosswalk_point2)
        else:
            frame2 = frame1
            crosswalk_point2 = None

        # 3. 차량 감지 (vehicle_detector 필요)
        if hasattr(self, 'vehicle_detector'):

            if int(len(crosswalk_point1)) > 0:
                frame3, self.warning = self.vehicle_detector.detect_vehicle(frame2, crosswalk_point1)
                #print("Crosswalk points1:", crosswalk_point1)
            elif int(len(crosswalk_point2)) > 0:
                frame3, self.warning = self.vehicle_detector.detect_vehicle(frame2, crosswalk_point2, self.traffic_status)
                #print("Crosswalk points2:", crosswalk_point2)
            else:
                frame3 = frame2
        else:
            frame3 = frame2

        # 디버깅
        #print(self.traffic_status)

        if self.warning is not None:
            diff = datetime.now() - self.lastUpdate
            if diff.total_seconds() > 1:
                if self.warning == "차량":
                    self.setText("Warning : A vehicle approaches.")  
                    print(self.warning)  
                else:
                    self.setText("Warning : Don't cross")   
                    print(self.warning)  
                self.lastUpdate = datetime.now()
                #self.play_sound()
                #print("소리나야함")
        # -------------------------------
        # QGraphicsView에 출력
        # -------------------------------
        rgb_frame = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qimg = QImage(rgb_frame.data, w, h, w*ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.pixmapItem.setPixmap(pixmap)
        self.graphicsScene.setSceneRect(self.pixmapItem.boundingRect())
        self.graphicsView.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
            
    def replaceViewer(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qimg = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.pixmapItem.setPixmap(pixmap)
    
    def play_sound(self):
        self.effect.play()

if __name__ == "__main__":
    app = MainWindow()
    app.run()
