from PyQt5 import QtWidgets
from .correlationmodel import CorrelationCoefficientsModel
from .correlation_ui import Ui_Form
import numpy as np
from typing import List

class CorrelationCoefficientsTable(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)

    def setMatrix(self, matrix:np.ndarray, names:List[str]):
        oldmodel = self.tableView.model()
        newmodel = CorrelationCoefficientsModel(matrix, names)
        self.tableView.setModel(newmodel)
        if oldmodel is not None:
            oldmodel.deleteLater()
