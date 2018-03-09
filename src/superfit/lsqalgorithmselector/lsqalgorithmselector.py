from PyQt5 import QtWidgets, QtCore
from .lsqalgorithmselector_ui import Ui_Form

class LSQAlgorithmSelector(QtWidgets.QWidget, Ui_Form):
    algorithmChanged = QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.algorithmComboBox.currentIndexChanged.connect(self.onParameterChanged)
        self.lossFunctionComboBox.currentIndexChanged.connect(self.onParameterChanged)
        self.jacobianComboBox.currentIndexChanged.connect(self.onParameterChanged)
        self.xtolDoubleSpinBox.valueChanged.connect(self.onParameterChanged)
        self.ftolDoubleSpinBox.valueChanged.connect(self.onParameterChanged)
        self.gtolDoubleSpinBox.valueChanged.connect(self.onParameterChanged)
        self.maxnfevSpinBox.valueChanged.connect(self.onParameterChanged)
        self.jacRescaleCheckBox.toggled.connect(self.onParameterChanged)
        self.weightWithErrorsCheckBox.toggled.connect(self.onParameterChanged)
        self.useLogYCheckBox.toggled.connect(self.onParameterChanged)
        self.xtolDoubleSpinBox.setValue(1e-8)
        self.ftolDoubleSpinBox.setValue(1e-8)
        self.gtolDoubleSpinBox.setValue(1e-8)

    def onParameterChanged(self):
        self.algorithmChanged.emit(self.algorithmKwargs())

    def algorithmKwargs(self):
        max_nfev = self.maxnfevSpinBox.value()
        if max_nfev == 0:
            max_nfev = None
        return {'method':self.algorithmComboBox.currentText(),
                  'loss':self.lossFunctionComboBox.currentText(),
                  'jac':self.jacobianComboBox.currentText(),
                  'xtol':self.xtolDoubleSpinBox.value(),
                  'ftol':self.ftolDoubleSpinBox.value(),
                  'gtol':self.gtolDoubleSpinBox.value(),
                  'max_nfev':max_nfev,
                  'x_scale':[1.0, 'jac'][self.jacRescaleCheckBox.isChecked()],
                  'dy_weight':self.weightWithErrorsCheckBox.isChecked(),
                  'logy':self.useLogYCheckBox.isChecked()
        }
