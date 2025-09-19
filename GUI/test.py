import sys, requests, numpy as np, cv2
from PySide6.QtWidgets import QApplication, QLCDNumber, QPushButton, QLabel, QFileDialog, QMessageBox
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import QFile, Qt, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QPixmap

class CrosswalkSign():
    def __init__(self, signWidget, LCDWidget, startCount=10, blinkInterval=500):
        self.sign = signWidget
        self.LCD = LCDWidget
        
        self.greenLightOnImg = QPixmap('GUI/img/greenLightOn.png')
        self.greenLightOffImg = QPixmap('GUI/img/greenLightOff.png')
        self.redLightImg = QPixmap('GUI/img/redLight.png')

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
        
        #어플리케이션 객체 생성 및 ui 파일 불러오기
        self.app = QApplication([])
        uiFile = QFile('GUI/mainWindow.ui')
        uiFile.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.window = loader.load(uiFile)
        uiFile.close()
        
        #GUI 위젯 생성 및 초기화

        self.signLabel = self.window.findChild(QLabel, "signLabel")
        self.countLCD = self.window.findChild(QLCDNumber, "countLCD")
        self.countLCD.setStyleSheet("color: #20b220;")
        self.countLCD.display('')

        self.testButton = self.window.findChild(QPushButton, "testButton")
        self.testButton.clicked.connect(self.testButtonAct)

        self.cctvButton = self.window.findChild(QPushButton, "cctvButton")
        self.cctvButton.clicked.connect(self.testCCTVLive)
        
        self.videoFileButton = self.window.findChild(QPushButton, "videoFileButton")
        self.videoFileButton.clicked.connect(self.openVideoFile)
        
        self.graphicsView = self.window.findChild(QGraphicsView, "graphicsView")
        self.graphicsScene = QGraphicsScene()
        self.graphicsView.setScene(self.graphicsScene)
        self.pixmapItem = QGraphicsPixmapItem()
        self.graphicsScene.addItem(self.pixmapItem)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)
            
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
            "비디오 파일을 선택하세요.",
            "./",
            "Videos (*.mp4 *.MP4 *.mkv *.MKV)"
        )
        if filePath:
            self.videoCap = cv2.VideoCapture(filePath)
            if not self.videoCap.isOpened():
                QMessageBox.warning(self.window, "에러", "비디오를 열 수 없습니다.")
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
        
        self.nextFrame()
        self.graphicsScene.setSceneRect(self.pixmapItem.boundingRect())
        self.graphicsView.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        fps = 60
        interval = int(1000 / fps) if fps > 0 else 33
        self.timer.start(interval)
    
    def nextFrame(self, ret = None, frame = None):
        if ret is None:
            ret, frame = self.videoCap.read()
        self.currentFrame += 1
        if self.currentFrame >= self.MaximumFrame:
            self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.currentFrame = -1
        if ret:
            self.replaceViewer(frame)
            
    def replaceViewer(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qimg = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.pixmapItem.setPixmap(pixmap)
    
if __name__ == "__main__":
    app = MainWindow()
    app.run()