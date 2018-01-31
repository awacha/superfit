from PyQt5 import QtWidgets
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from .graph_ui import Ui_Form
import numpy as np
from typing import Optional, List, Union

class Graph(QtWidgets.QWidget, Ui_Form):
    figure: Figure = None
    canvas: FigureCanvasQTAgg = None
    toolbar: NavigationToolbar2QT = None
    axes: Axes = None
    _data: np.ndarray = None
    _fitcurve: np.ndarray = None
    _dataline: Union[ErrorbarContainer, Line2D] = None
    _fitline: Line2D = None


    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        l = self.layout()
        assert isinstance(l, QtWidgets.QVBoxLayout)
        l.addWidget(self.canvas, stretch=1)
        l.addWidget(self.toolbar)
        self.replotDataPushButton.clicked.connect(self.replotData)
        self.replotModelPushButton.clicked.connect(self.replotModel)
        self.replotDataPushButton.setEnabled(False)
        self.replotModelPushButton.setEnabled(False)
        self.clf()

    def setDataSet(self, data:np.ndarray):
        self._data = data
        self.replotDataPushButton.setEnabled(True)

    def setFitCurve(self, curve:np.ndarray):
        self._fitcurve = curve
        self.replotModelPushButton.setEnabled(True)

    def replotData(self):
        if self._data is None:
            return
        self.clf()
        data = self._data
        x = data[:,0]
        y = data[:,1]
        try:
            dy = data[:,2]
        except IndexError:
            dy = None
        try:
            dx = data[:,3]
        except IndexError:
            dx = None
        if self.errorBarsCheckBox.isChecked():
            self._dataline = self.axes.errorbar(x, y, dy, dx, '.')
        self._updateGraphScales()
        self.canvas.draw()

    def _updateGraphScales(self):
        if self.plotTypeComboBox.currentText()=='Linear X vs. linear Y':
            self.axes.set_xscale('linear')
            self.axes.set_yscale('linear')
        elif self.plotTypeComboBox.currentText()=='Log X vs. log Y':
            self.axes.set_xscale('log')
            self.axes.set_yscale('log')
        elif self.plotTypeComboBox.currentText()=='Log X vs. linear Y':
            self.axes.set_xscale('log')
            self.axes.set_yscale('linear')
        elif self.plotTypeComboBox.currentText()=='Linear X vs. log Y':
            self.axes.set_xscale('linear')
            self.axes.set_yscale('log')
        else:
            raise ValueError('Unknown plot type: {}'.format(self.plotTypeComboBox.currentText()))

    def clf(self):
        self.figure.clear(keep_observers=False)
        self.axes = self.figure.add_subplot(1,1,1)
        self._dataline = None
        self._fitline = None
        self.canvas.draw()

    def replotModel(self):
        if self._fitcurve is None:
            return
        try:
            self._fitline.remove()
        except AttributeError:
            pass
        self._fitline = None
        self._fitline = self.axes.plot(self._fitcurve[:,0], self._fitcurve[:,1], 'r-')[0]
        self._updateGraphScales()
        self.axes.relim()
        self.axes.autoscale(True, True, True)
        self.canvas.draw()


