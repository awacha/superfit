from typing import List, Any, Union
import copy

import numpy as np
from PyQt5 import QtCore, QtGui

class Parameter:
    name:str=''
    value:float=0.0
    uncertainty:float=0.0
    fittable:bool=True
    fixed:bool=False
    lbound:float=-np.inf
    ubound:float=np.inf
    lbound_enabled:bool=True
    ubound_enabled:bool=True
    lbound_reached:bool=False
    ubound_reached:bool=False
    
    def __init__(self, name:str, value:float=0.0, uncertainty:float=0.0, 
                 lbound:float=-np.inf, ubound:float=np.inf, fittable:bool=True, fixed:bool=False,
                 lbound_enabled:bool=False, ubound_enabled:bool=False, lbound_reached:bool=False,
                 ubound_reached:bool=False):
        self.name=name
        self.value=value
        self.uncertainty=uncertainty
        self.lbound=lbound
        self.ubound=ubound
        self.fittable=fittable
        self.fixed=fixed
        self.lbound_enabled=lbound_enabled
        self.ubound_enabled=ubound_enabled
        self.lbound_reached=lbound_reached
        self.ubound_reached=ubound_reached
        
    
class ParameterModel(QtCore.QAbstractItemModel):
    # columns:
    #     name (checkable)
    #     lower bound (checkable, editable if checked)
    #     upper bound (checkable, editable if checked)
    #     value (editable)
    #     uncertainty
    #     relative uncertainty
    historyChanged = QtCore.pyqtSignal()

    _data:List[Parameter] = []
    _history: List[List[Parameter]] = []
    _history_index : int

    def __init__(self):
        super().__init__(None)
        self._data=[]
        self.historyReset()

    def cleanup(self):
        self.beginResetModel()
        self._data = []
        self._history = []
        self.endResetModel()
        self.deleteLater()

    def addParameter(self, name:str, value:float, lbound:float, ubound:float, fittable:bool, lbound_enabled:bool=False,
                     ubound_enabled:bool=False):
        self.beginInsertRows(QtCore.QModelIndex(), len(self._data), len(self._data))
        self._data.append(Parameter(name, value, 0, lbound, ubound, fittable, lbound_enabled=lbound_enabled,
                                    ubound_enabled=ubound_enabled))
        self.endInsertRows()

    def parameter(self, index:int) -> Parameter:
        return self._data[index]

    def _nametoindex(self, index:Union[int, str]) -> int:
        if isinstance(index, str):
            return [i for i in range(len(self._data)) if self._data[i].name==index][0]
        else:
            return index

    def changeValue(self, index:Union[int, str], newvalue:float):
        index = self._nametoindex(index)
        self.parameter(index).value = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()))

    def changeUncertainty(self, index:Union[int, str], newvalue:float):
        index = self._nametoindex(index)
        self.parameter(index).uncertainty = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()))

    def changeLBound(self, index: Union[int, str], newvalue: float):
        index = self._nametoindex(index)
        self.parameter(index).lbound = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()))

    def changeUBound(self, index: Union[int, str], newvalue: float):
        index = self._nametoindex(index)
        self.parameter(index).ubound = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()))

    def changeLBoundActive(self, index:Union[int, str], newvalue:bool):
        index = self._nametoindex(index)
        self.parameter(index).lbound_reached = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()))

    def changeUBoundActive(self, index:Union[int, str], newvalue:bool):
        index = self._nametoindex(index)
        self.parameter(index).ubound_reached = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()))


    def columnCount(self, parent: QtCore.QModelIndex = ...):
        return 6
    
    def rowCount(self, parent: QtCore.QModelIndex = ...):
        return len(self._data)
    
    def flags(self, index: QtCore.QModelIndex):
        flags=QtCore.Qt.ItemNeverHasChildren | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if index.column() in [0,1,2]: # name, lbound, ubound are checkable
            flags |= QtCore.Qt.ItemIsUserCheckable
        if index.column()==1 and self._data[index.row()].lbound_enabled:
            flags |= QtCore.Qt.ItemIsEditable
        if index.column()==2 and self._data[index.row()].ubound_enabled:
            flags |= QtCore.Qt.ItemIsEditable
        if index.column()==3: # value
            flags |= QtCore.Qt.ItemIsEditable
        return flags

    def data(self, index: QtCore.QModelIndex, role: int = ...):
        if role == QtCore.Qt.CheckStateRole:
            if index.column()==0:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][not self._data[index.row()].fixed]
            elif index.column()==1:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][self._data[index.row()].lbound_enabled]
            elif index.column()==2:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][self._data[index.row()].ubound_enabled]
            return None
        elif role == QtCore.Qt.DisplayRole:
            if index.column()==0:
                return self._data[index.row()].name
            elif index.column()==1:
                if self._data[index.row()].lbound_enabled:
                    return self._data[index.row()].lbound
                else:
                    return 'Free'
            elif index.column()==2:
                if self._data[index.row()].ubound_enabled:
                    return self._data[index.row()].ubound
                else:
                    return 'Free'
            elif index.column()==3:
                return str(self._data[index.row()].value)
            elif index.column()==4:
                if self._data[index.row()].fittable:
                    return str(self._data[index.row()].uncertainty)
                else:
                    return 'N/A'
            elif index.column()==5:
                val = abs(self._data[index.row()].value)
                unc = self._data[index.row()].uncertainty
                if unc==0:
                    return 'N/A'
                elif val==0:
                    return 'inf'
                else:
                    return '{:.0f} %'.format(unc/val*100)
            else:
                return None
        elif role == QtCore.Qt.EditRole:
            if index.column()==1:
                return self._data[index.row()].lbound
            elif index.column()==2:
                return self._data[index.row()].ubound
            elif index.column()==3:
                return self._data[index.row()].value
        elif role == QtCore.Qt.BackgroundColorRole:
            if index.column()==1 and self._data[index.row()].lbound_enabled and self._data[index.row()].value<self._data[index.row()].lbound:
                return QtGui.QColor('red')
            elif index.column()==2 and self._data[index.row()].ubound_enabled and self._data[index.row()].value>self._data[index.row()].ubound:
                return QtGui.QColor('red')
            elif index.column()==1 and self._data[index.row()].lbound_reached:
                return QtGui.QColor('orange')
            elif index.column()==2 and self._data[index.row()].ubound_reached:
                return QtGui.QColor('orange')
            elif index.column()==5:
                val = abs(self._data[index.row()].value)
                unc = self._data[index.row()].uncertainty
                if unc>0.5*val:
                    return QtGui.QColor('orange')
                elif unc>val:
                    return QtGui.QColor('red')
                else:
                    return None
                    #return QtGui.QColor('green')
            return None
        return None

    def setData(self, index: QtCore.QModelIndex, value: Any, role: int = ...):
        if role == QtCore.Qt.CheckStateRole:
            if index.column()==0:
                self._data[index.row()].fixed = not value==QtCore.Qt.Checked
                self.dataChanged.emit(self.index(index.row(), index.column(), QtCore.QModelIndex()),
                                      self.index(index.row(), index.column(), QtCore.QModelIndex()))
                return True
            elif index.column() == 1:
                self._data[index.row()].lbound_enabled = value==QtCore.Qt.Checked
                self.dataChanged.emit(self.index(index.row(), index.column(), QtCore.QModelIndex()),
                                      self.index(index.row(), index.column(), QtCore.QModelIndex()))
                return True
            elif index.column() == 2:
                self._data[index.row()].ubound_enabled = value == QtCore.Qt.Checked
                self.dataChanged.emit(self.index(index.row(), index.column(), QtCore.QModelIndex()),
                                      self.index(index.row(), index.column(), QtCore.QModelIndex()))
                return True
            else:
                return False
        elif role == QtCore.Qt.EditRole:
            try:
                value = float(value)
            except ValueError:
                return False
            if index.column()==1:
                self._data[index.row()].lbound = value
                self.dataChanged.emit(self.index(index.row(), 0, QtCore.QModelIndex()),
                                      self.index(index.row(), self.columnCount(), QtCore.QModelIndex()))
                return True
            elif index.column()==2:
                self._data[index.row()].ubound = value
                self.dataChanged.emit(self.index(index.row(), 0, QtCore.QModelIndex()),
                                      self.index(index.row(), self.columnCount(), QtCore.QModelIndex()))
                return True
            elif index.column()==3:
                self._data[index.row()].value = value
                self.dataChanged.emit(self.index(index.row(), 0, QtCore.QModelIndex()),
                                      self.index(index.row(), self.columnCount(), QtCore.QModelIndex()))
                return True
            return False
        return False

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...):
        if orientation == QtCore.Qt.Horizontal and role==QtCore.Qt.DisplayRole:
            return ['Name', 'Left limit', 'Right limit', 'Value', 'Uncertainty', 'Rel. uncertainty'][section]
        else:
            return None
    
    def parent(self, child: QtCore.QModelIndex):
        return QtCore.QModelIndex()
    
    def index(self, row: int, column: int, parent: QtCore.QModelIndex = ...):
        return self.createIndex(row, column, None)

    def historyBack(self):
        self.historyGoto(self._history_index-1)

    def historyForward(self):
        self.historyGoto(self._history_index+1)

    def historyPush(self):
        self._history = self._history[:self._history_index+1]
        self._history.append(copy.deepcopy(self._data))
        self._history_index+=1
        self.historyChanged.emit()

    def historyGoto(self, index:int):
        if index <0 or index > len(self._history)-1:
            return
        self._history_index = index
        self.beginResetModel()
        self._data = copy.deepcopy(self._history[self._history_index])
        self.endResetModel()
        self.historyChanged.emit()

    def historyReset(self):
        self._history = []
        self._history_index = -1
        self.historyChanged.emit()

    def historyIndex(self):
        return self._history_index

    def historySize(self):
        return len(self._history)

    def __iter__(self):
        for p in self._data:
            yield p