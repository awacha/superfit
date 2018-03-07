from typing import List, Any, Union
import copy
import pickle

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

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
    description:str=''
    
    def __init__(self, name:str, value:float=0.0, uncertainty:float=0.0, 
                 lbound:float=-np.inf, ubound:float=np.inf, fittable:bool=True, fixed:bool=False,
                 lbound_enabled:bool=False, ubound_enabled:bool=False, lbound_reached:bool=False,
                 ubound_reached:bool=False, description:str=''):
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
        self.description=description

    def toDict(self):
        return {'name':self.name,
                'value':self.value,
                'uncertainty':self.uncertainty,
                'lbound':self.lbound,
                'ubound':self.ubound,
                'fittable':self.fittable,
                'fixed':self.fixed,
                'lbound_enabled':self.lbound_enabled,
                'lbound_reached':self.lbound_reached,
                'ubound_enabled':self.ubound_enabled,
                'ubound_reached':self.ubound_reached,
                }

    def fromDict(self, dict):
        self.name = dict['name']
        self.value = dict['value']
        self.uncertainty=dict['uncertainty']
        self.lbound=dict['lbound']
        self.ubound=dict['ubound']
        self.fittable=dict['fittable']
        self.fixed=dict['fixed']
        self.lbound_enabled=dict['lbound_enabled']
        self.lbound_reached=dict['lbound_reached']
        self.ubound_enabled = dict['ubound_enabled']
        self.ubound_reached = dict['ubound_reached']
    
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

    def saveState(self, filename):
        lis = [p.toDict() for p in self._data]
        with open(filename, 'wb') as f:
            pickle.dump(lis, f)

    def loadState(self, filename):
        with open(filename, 'rb') as f:
            data=pickle.load(f)
        names = sorted([p['name'] for p in data])
        if names != [p.name for p in self._data]:
            raise ValueError('Incompatible parameter data file')
        for p in self._data:
            dic=[d for d in data if d['name'] == p.name][0]
            p.fromDict(dic)
        self.dataChanged.emit(self.index(0,0), self.index(self.rowCount(), self.columnCount()),
                              [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.BackgroundColorRole, QtCore.Qt.CheckStateRole])
        self.historyPush()

    def addParameter(self, name:str, value:float, lbound:float, ubound:float, fittable:bool, lbound_enabled:bool=False,
                     ubound_enabled:bool=False, description:str=''):
        self.beginInsertRows(QtCore.QModelIndex(), len(self._data), len(self._data))
        self._data.append(Parameter(name, value, 0, lbound, ubound, fittable, lbound_enabled=lbound_enabled,
                                    ubound_enabled=ubound_enabled, description=description))
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
        self.dataChanged.emit(self.index(index, 3), self.index(index, self.columnCount()), [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.BackgroundColorRole])

    def changeUncertainty(self, index:Union[int, str], newvalue:float):
        index = self._nametoindex(index)
        self.parameter(index).uncertainty = newvalue
        self.dataChanged.emit(self.index(index, 3), self.index(index, self.columnCount()), [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.BackgroundColorRole])

    def changeLBound(self, index: Union[int, str], newvalue: float):
        index = self._nametoindex(index)
        self.parameter(index).lbound = newvalue
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.BackgroundColorRole])

    def changeUBound(self, index: Union[int, str], newvalue: float):
        index = self._nametoindex(index)
        self.parameter(index).ubound = newvalue
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.BackgroundColorRole])

    def changeLBoundActive(self, index:Union[int, str], newvalue:bool):
        index = self._nametoindex(index)
        self.parameter(index).lbound_reached = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()), [QtCore.Qt.CheckStateRole])
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.BackgroundColorRole])

    def changeUBoundActive(self, index:Union[int, str], newvalue:bool):
        index = self._nametoindex(index)
        self.parameter(index).ubound_reached = newvalue
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount()), [QtCore.Qt.CheckStateRole])
        self.dataChanged.emit(self.index(index, 1), self.index(index, self.columnCount()), [QtCore.Qt.BackgroundColorRole])


    def columnCount(self, parent: QtCore.QModelIndex = ...):
        return 6
    
    def rowCount(self, parent: QtCore.QModelIndex = ...):
        return len(self._data)
    
    def flags(self, index: QtCore.QModelIndex):
        flags=QtCore.Qt.ItemNeverHasChildren | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if index.column() in [0,1,2] and self._data[index.row()].fittable: # name, lbound, ubound are checkable
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
                if not self._data[index.row()].fittable:
                    return None
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
        elif role == QtCore.Qt.ToolTipRole:
            return self._data[index.row()].description
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
                                      self.index(index.row(), self.columnCount(), QtCore.QModelIndex()),
                                      [QtCore.Qt.EditRole, QtCore.Qt.DisplayRole])
                return True
            elif index.column()==2:
                self._data[index.row()].ubound = value
                self.dataChanged.emit(self.index(index.row(), 0, QtCore.QModelIndex()),
                                      self.index(index.row(), self.columnCount(), QtCore.QModelIndex()),
                                      [QtCore.Qt.EditRole, QtCore.Qt.DisplayRole])
                return True
            elif index.column()==3:
                self._data[index.row()].value = value
                self.dataChanged.emit(self.index(index.row(), 0, QtCore.QModelIndex()),
                                      self.index(index.row(), self.columnCount(), QtCore.QModelIndex()),
                                      [QtCore.Qt.EditRole, QtCore.Qt.DisplayRole])
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

class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent: QtWidgets.QWidget, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        editor = QtWidgets.QDoubleSpinBox(parent)
        editor.setMinimum(-10e20)
        editor.setMaximum(10e20)
        editor.setDecimals(16)
        return editor

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex):
        value = index.data(QtCore.Qt.EditRole)
        assert isinstance(editor, QtWidgets.QDoubleSpinBox)
        editor.setValue(value)

    def updateEditorGeometry(self, editor: QtWidgets.QWidget, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        editor.setGeometry(option.rect)

    def setModelData(self, editor: QtWidgets.QWidget, model: QtCore.QAbstractItemModel, index: QtCore.QModelIndex):
        model.setData(index, editor.value(), QtCore.Qt.EditRole)


