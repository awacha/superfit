from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import fffs
from .mainwindow_ui import Ui_MainWindow
from ..datasetselector import DataSetSelector
from ..parameters import ParameterView
from ..modelselector import ModelSelector
from ..lsqalgorithmselector import LSQAlgorithmSelector
from ..graph import Graph

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _data: np.ndarray=None
    _model: fffs.ModelFunction=None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pagewidgets = []
        self.setupUi(self)
        self._data = None

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.toolBox.removeItem(0)
        del self.page
        self.dataselector = DataSetSelector(self)
        self.modelselector = ModelSelector(self)
        self.parameterview = ParameterView(self)
        self.lsqalgorithmselector = LSQAlgorithmSelector(self)
        for text, attrname in [
            ('Load data...', 'dataselector'),
            ('Model function', 'modelselector'),
            ('Algorithm tuning', 'lsqalgorithmselector'),
            ('Parameters && fitting...', 'parameterview')]:
            w=QtWidgets.QWidget(self.toolBox)
            w.setContentsMargins(0,0,0,0)
            l=QtWidgets.QVBoxLayout(w)
            l.setContentsMargins(0,0,0,0)
            w.setLayout(l)
            l.addWidget(getattr(self, attrname))
            l.addStretch(1)
            self._pagewidgets.append(w)
            self.toolBox.addItem(self._pagewidgets[-1], text)
        self.dataselector.dataSetLoaded.connect(self.onDataLoaded)
        self.modelselector.modelSelected.connect(self.onModelSelected)
        self.parameterview.fitCurveChanged.connect(self.onFitCurveChanged)
        self.tabWidget.removeTab(0)
        self.graph = Graph(self)
        self.tabWidget.addTab(self.graph,'Data && fit')
#        self.tabWidget.addTab(..., 'Model representation')
#        self.tabWidget.addTab(..., 'Parameter correlation')
#        self.tabWidget.addTab(..., 'Result && statistics')
        self.onModelSelected(self.modelselector.model())

    def onDataLoaded(self, data:np.ndarray):
        self._data = data
        self.graph.setDataSet(self._data)
        self.parameterview.setDataSet(self._data)
        self.graph.replotData()

    def onModelSelected(self, model:fffs.ModelFunction):
        self._model = model
        self.parameterview.setModelFunc(self._model)

    def plotData(self):
        self.graph.replotData()

    def plotFitted(self):
        self.graph.replotModel()

    def onFitCurveChanged(self, fitcurve:np.ndarray):
        self.graph.setFitCurve(fitcurve)
        self.graph.replotModel()