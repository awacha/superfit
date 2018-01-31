from PyQt5 import QtWidgets, QtCore
import fffs
from .modelselector_ui import Ui_Form

class ModelSelector(QtWidgets.QWidget, Ui_Form):
    modelSelected = QtCore.pyqtSignal(fffs.ModelFunction)
    _model: fffs.ModelFunction = None

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.categoryComboBox.currentIndexChanged.connect(self.onCategoryChanged)
        self.subCategoryComboBox.currentIndexChanged.connect(self.onSubCategoryChanged)
        self.nameComboBox.currentIndexChanged.connect(self.onNameSelected)
        self.categoryComboBox.addItems(fffs.categories())

    def onCategoryChanged(self):
        try:
            self.subCategoryComboBox.blockSignals(True)
            self.subCategoryComboBox.clear()
        finally:
            self.subCategoryComboBox.blockSignals(False)
        self.subCategoryComboBox.addItems(fffs.subcategories(self.categoryComboBox.currentText()))
        self.subCategoryComboBox.setEnabled(True)

    def onSubCategoryChanged(self):
        try:
            self.nameComboBox.blockSignals(True)
            self.nameComboBox.clear()
        finally:
            self.nameComboBox.blockSignals(False)
        self.nameComboBox.addItems(fffs.models(self.categoryComboBox.currentText(),
                                               self.subCategoryComboBox.currentText()))
        self.nameComboBox.setEnabled(True)

    def onNameSelected(self):
        modeltype = fffs.model(self.categoryComboBox.currentText(),
                           self.subCategoryComboBox.currentText(),
                           self.nameComboBox.currentText())
        self._model = modeltype()
        self.descriptionTextBrowser.setPlainText(self._model.description)
        self.modelSelected.emit(self._model)

    def model(self):
        return self._model