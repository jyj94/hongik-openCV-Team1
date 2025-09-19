import sys, requests
from PySide6.QtWidgets import QApplication, QLCDNumber, QPushButton, QLabel
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QPixmap

class MainWindow():
    def __init__(self):
        #변수 선언
        self.lightIsGreen = True
        
        #어플리케이션 객체 생성 및 ui 파일 불러오기
        self.app = QApplication([])
        uiFile = QFile('GUI/mainWindow.ui')
        uiFile.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.window = loader.load(uiFile)
        uiFile.close()
        
        #GUI 위젯 생성 및 초기화
        self.greenLightImg = QPixmap('GUI/img/greenLight.png')
        self.redLightImg = QPixmap('GUI/img/redLight.png')

        self.signLabel = self.window.findChild(QLabel, "signLabel")
        self.signLabel.setPixmap(self.greenLightImg)
        self.testButton = self.window.findChild(QPushButton, "testButton")
        self.testButton.clicked.connect(self.testButtonAct)

        self.countLCD = self.window.findChild(QLCDNumber, "countLCD")
        self.countLCD.setStyleSheet("color: #20b220;")
        self.countNum = 0
        
    def changeSign(self):
        if self.lightIsGreen:
            self.signLabel.setPixmap(self.redLightImg)
            self.lightIsGreen = False
        else:
            self.signLabel.setPixmap(self.greenLightImg)
            self.lightIsGreen = True
            
    def setLCDValue(self, input):
        self.countLCD.display(int(input))
        
    def run(self):
        self.window.show()
        sys.exit(self.app.exec())
        

    def testButtonAct(self):
        self.changeSign()
        self.countNum += 1
        self.setLCDValue(self.countNum)
 
if __name__ == "__main__":
    app = MainWindow()
    app.run()