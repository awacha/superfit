from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import fffs
from typing import Dict
from .mainwindow_ui import Ui_MainWindow
from ..datasetselector import DataSetSelector
from ..parameters import ParameterView, FitResults
from ..modelselector import ModelSelector
from ..lsqalgorithmselector import LSQAlgorithmSelector
from ..correlation import CorrelationCoefficientsTable
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
        self.correlationcoefficients = CorrelationCoefficientsTable(self)
        for text, attrname in [
            ('Load data...', 'dataselector'),
            ('Model function', 'modelselector'),
            ('Algorithm tuning', 'lsqalgorithmselector'),
            ('Parameters && fitting...', 'parameterview')]:
            w=QtWidgets.QWidget(self.toolBox)
            w.setContentsMargins(0,0,0,0)
            w.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
            l=QtWidgets.QVBoxLayout(w)
            l.setContentsMargins(0,0,0,0)
            w.setLayout(l)
            l.addWidget(getattr(self, attrname))
            #l.addStretch(0)
            self._pagewidgets.append(w)
            self.toolBox.addItem(self._pagewidgets[-1], text)
        self.dataselector.dataSetLoaded.connect(self.onDataLoaded)
        self.modelselector.modelSelected.connect(self.onModelSelected)
        self.parameterview.fitCurveChanged.connect(self.onFitCurveChanged)
        self.parameterview.fitResultsReady.connect(self.onFitResultsReady)
        self.lsqalgorithmselector.algorithmChanged.connect(self.onAlgorithmChanged)
        self.tabWidget.removeTab(0)
        self.graph = Graph(self)
        self.tabWidget.addTab(self.graph,'Data && fit')
#        self.tabWidget.addTab(..., 'Model representation')
        self.tabWidget.addTab(self.correlationcoefficients, 'Parameter correlation')
#        self.tabWidget.addTab(..., 'Result && statistics')
        self.onModelSelected(self.modelselector.model())
        self.onAlgorithmChanged(self.lsqalgorithmselector.algorithmKwargs())

    def onFitResultsReady(self, fitresults:FitResults):
        self.correlationcoefficients.setMatrix(
            fitresults.correl_coeffs,
            [p for p, f in zip(fitresults.paramnames, fitresults.paramfree) if f])

    def onDataLoaded(self, data:np.ndarray):
        self._data = data
        self.graph.setDataSet(self._data)
        self.parameterview.setDataSet(self._data)
        self.graph.replotData()

    def onModelSelected(self, model:fffs.ModelFunction):
        self._model = model
        self.parameterview.setModelFunc(self._model)

    def onAlgorithmChanged(self, kwargs:Dict):
        self.parameterview.setAlgorithmKwargs(kwargs)

    def plotData(self):
        self.graph.replotData()

    def plotFitted(self):
        self.graph.replotModel()

    def onFitCurveChanged(self, fitcurve:np.ndarray):
        self.graph.setFitCurve(fitcurve)
        self.graph.replotModel()