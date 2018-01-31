import numpy as np
from PyQt5 import QtCore, QtWidgets
from .datasetselector_ui import Ui_Form

class DataSetSelector(QtWidgets.QWidget, Ui_Form):
    dataSetLoaded = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset=None
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.browsePushButton.clicked.connect(self.onBrowse)
        self.reloadPushButton.clicked.connect(self.onReload)
        self.xMinDoubleSpinBox.valueChanged.connect(self.onLimitsChanged)
        self.xMaxDoubleSpinBox.valueChanged.connect(self.onLimitsChanged)
        self.reloadPushButton.setEnabled(False)
        self.xMinDoubleSpinBox.setEnabled(False)
        self.xMaxDoubleSpinBox.setEnabled(False)

    def onLimitsChanged(self):
        self.xMaxDoubleSpinBox.setMinimum(self.xMinDoubleSpinBox.value())
        self.xMinDoubleSpinBox.setMaximum(self.xMaxDoubleSpinBox.value())
        self.onReload()

    def onBrowse(self):
        fname, fltr = QtWidgets.QFileDialog.getOpenFileName(self,'Open data file','')
        if not fname:
            return
        self.xMinDoubleSpinBox.setEnabled(True)
        self.xMaxDoubleSpinBox.setEnabled(True)
        self.reloadPushButton.setEnabled(True)
        self.fileNameLineEdit.setText(fname)
        self.onReload()

    def onReload(self):
        try:
            self._dataset = np.loadtxt(self.fileNameLineEdit.text())
            idx = np.prod(np.isfinite(self._dataset), axis=1).astype(np.bool)
            self._dataset = self._dataset[idx]
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Cannot open file', 'Error while opening file {}: {}'.format(self.fileNameLineEdit.text(), exc.args[0]))
            return
        if self.windowFilePath() != self.fileNameLineEdit.text():
            # a new file has been loaded
            self.xMinDoubleSpinBox.setMinimum(np.nanmin(self._dataset[:,0]))
            self.xMinDoubleSpinBox.setMaximum(np.nanmax(self._dataset[:,0]))
            self.xMaxDoubleSpinBox.setMinimum(np.nanmin(self._dataset[:,0]))
            self.xMaxDoubleSpinBox.setMaximum(np.nanmax(self._dataset[:, 0]))
            self.xMinDoubleSpinBox.setValue(np.nanmin(self._dataset[:,0]))
            self.xMaxDoubleSpinBox.setValue(np.nanmax(self._dataset[:,0]))
            self.setWindowFilePath(self.fileNameLineEdit.text())
        self.dataSetLoaded.emit(self.dataSet())


    def dataSet(self):
        idx = np.logical_and(self._dataset[:,0]<=self.xMaxDoubleSpinBox.value(),
                             self._dataset[:,0]>=self.xMinDoubleSpinBox.value())
        return self._dataset[idx,:]


