from PySide6.QtWidgets import QMessageBox, QApplication
from PyQt5.QtMultimedia import QSound

app = QApplication()
QSound.play('GUI\경고음.mp3')
mssage = QMessageBox.warning(None, '경고음 재생', '경고음 재생', QMessageBox.Ok)

app.exec()
mssage.exec()
