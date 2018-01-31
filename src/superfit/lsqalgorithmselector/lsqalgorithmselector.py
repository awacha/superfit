from PyQt5 import QtWidgets
from .lsqalgorithmselector_ui import Ui_Form

class LSQAlgorithmSelector(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)

