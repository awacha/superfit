from PyQt5 import QtCore, QtGui
import numpy as np
from typing import List


class CorrelationCoefficientsModel(QtCore.QAbstractItemModel):
    def __init__(self, matrix:np.ndarray, names:List[str]):
        super().__init__()
        self._matrix = matrix
        self._names = names
        assert matrix.shape[0] == matrix.shape[1]
        assert len(names) == matrix.shape[0]

    def columnCount(self, parent: QtCore.QModelIndex = ...):
        return len(self._names)

    def rowCount(self, parent: QtCore.QModelIndex = ...):
        return len(self._names)

    def flags(self, index: QtCore.QModelIndex):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemNeverHasChildren | QtCore.Qt.ItemIsSelectable

    def data(self, index: QtCore.QModelIndex, role: int = ...):
        if role == QtCore.Qt.DisplayRole:
            return '{:.3f}'.format(self._matrix[index.row(), index.column()])
        if role == QtCore.Qt.BackgroundColorRole:
            if index.column() == index.row():
                return QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 1.0)
            else:
                val = 1-min(1,max(0,abs(self._matrix[index.row(), index.column()])))
                return QtGui.QColor.fromRgbF(1.0, val, val, 1.0)
        return None

    def index(self, row: int, column: int, parent: QtCore.QModelIndex = ...):
        return self.createIndex(row, column, None)

    def parent(self, child: QtCore.QModelIndex):
        return QtCore.QModelIndex()

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...):
        if role == QtCore.Qt.DisplayRole:
            return self._names[section]

    def deleteLater(self):
        del self._matrix
        self._matrix = None
        del self._names
        self._names = None
        return super().deleteLater()