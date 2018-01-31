from PyQt5 import QtWidgets
import sys
from .mainwindow import MainWindow

def run():
    app = QtWidgets.QApplication(sys.argv)
    mainwin = MainWindow()
    mainwin.show()
    result = app.exec_()
    mainwin.deleteLater()
    del mainwin
    app.deleteLater()
    del app
    sys.exit(result)

if __name__=='__main__':
    run()