from PyQt5 import QtWidgets, QtCore
import fffs
import scipy.optimize
import traceback
from scipy.linalg import svd
from .parametermodel import ParameterModel, Parameter
from .parameters_ui import Ui_Form
import sys
import numpy as np
from typing import Sequence, Callable, Type, List

class FitResults:
    dataset:np.ndarray=None
    funcvalue:np.ndarray=None
    paramnames:List[str]=None
    paramvalues:np.ndarray=None
    paramuncertainties:np.ndarray=None
    paramfree:np.ndarray=None
    paramlbounds:np.ndarray=None
    paramubounds:np.ndarray=None
    paramlboundactive:np.ndarray=None
    paramuboundactive:np.ndarray=None
    paramactive_mask:np.ndarray=None
    covar:np.ndarray=None
    correl_coeffs:np.ndarray=None
    R2:float=None
    R2_adj:float=None
    chi2:float=None
    chi2_reduced:float=None
    dof:int=None


class FitWorker(QtCore.QThread):
    resultsReady = QtCore.pyqtSignal(scipy.optimize.OptimizeResult)
    exceptionOccurred = QtCore.pyqtSignal(type, Exception, object)

    def __init__(self, function:Callable, dataset:np.ndarray, param_init:Sequence,
                 fittable:Sequence, lbounds:Sequence, ubounds:Sequence, **kwargs):
        super().__init__(None)
        self.free = np.array(fittable, dtype=np.bool)
        self.param_init = np.array(param_init, dtype=np.double)
        self.x = dataset[:,0]
        self.y = dataset[:,1]
        try:
            self.dy = dataset[:,2]
        except IndexError:
            self.dy = np.ones_like(param_init)
        try:
            self.dx = dataset[:,3]
        except IndexError:
            self.dx = np.ones_like(param_init)
        self.function = function
        self.lbounds = np.array(lbounds)
        self.ubounds = np.array(ubounds)
        self.lsq_kwargs = kwargs

    def fitfunc(self, params:np.ndarray):
        self.param_init[self.free] = params
        return (self.y-self.function(self.x, *self.param_init))/self.dy

    def run(self):
        try:
            result = scipy.optimize.least_squares(
                self.fitfunc, self.param_init[self.free],
                bounds=(self.lbounds[self.free], self.ubounds[self.free]),
                **self.lsq_kwargs
            )
            self.resultsReady.emit(result)
        except Exception:
            self.exceptionOccurred.emit(*sys.exc_info())


class ParameterView(QtWidgets.QWidget, Ui_Form):
    func: fffs.ModelFunction = None
    fitcurve: np.ndarray = None
    fitCurveChanged = QtCore.pyqtSignal(np.ndarray)
    dataset: np.ndarray = None
    fitworker: FitWorker = None

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.model = ParameterModel()
        self.model.dataChanged.connect(self.onDataChanged)
        self.model.historyChanged.connect(self.onHistoryChanged)
        self.treeView.setModel(self.model)
        self.m5ToolButton.clicked.connect(self.onMinus5)
        self.p5ToolButton.clicked.connect(self.onPlus5)
        self.m20ToolButton.clicked.connect(self.onMinus20)
        self.p20ToolButton.clicked.connect(self.onPlus20)
        self.resetPushButton.clicked.connect(self.onResetParameters)
        self.treeView.selectionModel().currentRowChanged.connect(self.onSelectionChanged)
        self.stepPushButton.clicked.connect(self.onStep)
        self.fitPushButton.clicked.connect(self.onFit)
        self.onSelectionChanged()

    def onStep(self):
        pass

    def onFit(self):
        if self.fitworker is not None:
            return
        self.fitworker = FitWorker(
            self.func.fitfunction, self.dataset,
            [self.model.parameter(i).value for i in range(self.model.rowCount())],
            [self.model.parameter(i).fittable and not self.model.parameter(i).fixed
             for i in range(self.model.rowCount())],
            [(-np.inf,self.model.parameter(i).lbound)[self.model.parameter(i).lbound_enabled]
             for i in range(self.model.rowCount())],
            [(np.inf, self.model.parameter(i).ubound)[self.model.parameter(i).lbound_enabled]
             for i in range(self.model.rowCount())]
        )
        self.fitworker.resultsReady.connect(self.onFitResultsReady)
        self.fitworker.exceptionOccurred.connect(self.onFitException)
        self.fitworker.finished.connect(self.onFitFinished)
        self.fitworker.started.connect(self.onFitStarted)
        self.fitworker.start()

    def onFitStarted(self):
        pass

    def onFitResultsReady(self, results:scipy.optimize.OptimizeResult):
        param_fitted = np.array([self.model.parameter(i).value for i in range(self.model.rowCount())])
        free = np.array([self.model.parameter(i).fittable and not self.model.parameter(i).fixed
                         for i in range(self.model.rowCount())])
        param_fitted[free] = results.x
        funcvalue = self.func.fitfunction(self.dataset[:,0], *param_fitted)
        _, s, VT = svd(results.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(results.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        #ToDo: other statistics like chi^2, R^2
        uncertainties = np.zeros_like(param_fitted)
        uncertainties[free] = np.diag(pcov)**0.5
        active_mask = np.zeros(len(param_fitted), dtype=np.uint8)
        active_mask[free] = results.active_mask
        correl_coeffs = pcov / np.outer(np.diag(pcov)**0.5, np.diag(pcov)**0.5)
        chi2 = (results.fun**2).sum()
        dof = self.dataset.shape[0]-free.sum()
        chi2_red = chi2/dof
        sstot = ((self.dataset[:,1]-np.mean(self.dataset[:,1]))**2).sum()
        ssres = ((funcvalue - self.dataset[:,1])**2).sum()
        R2 = 1-ssres/sstot
        R2_adj = 1 - (ssres/ (dof-1)) / (sstot/(self.dataset.shape[0]-1))
        if self.dataset.shape[1]>2:
            sstot_w = (((self.dataset[:, 1] - np.mean(self.dataset[:, 1]))/self.dataset[:,2]) ** 2).sum()
            ssres_w = (((funcvalue - self.dataset[:, 1])/self.dataset[:,2]) ** 2).sum()
            R2_w = 1-ssres_w/sstot_w
            R2_adj_w = 1- (ssres_w / (dof-1)) / (sstot_w/(self.dataset.shape[0]-1))
        else:
            sstot_w = sstot
            ssres_w = ssres
            R2_w = R2
            R2_adj_w = R2_adj

        for i in range(self.model.rowCount()):
            if free[i]:
                self.model.changeValue(i, param_fitted[i])
                self.model.changeUncertainty(i, uncertainties[i])
                print(i, param_fitted[i], uncertainties[i])
            else:
                self.model.changeUncertainty(i, 0)
        self.model.historyPush()
        self.onNewFitCurve()

    def onFitException(self, exctype:Type[Exception], exc:Exception, tb:{'tb_frame', 'tb_lineno'}):
        QtWidgets.QMessageBox.critical(self, 'Error while fitting', '\n'.join(traceback.format_exception(exctype, exc, tb)))

    def onFitFinished(self):
        self.fitworker.deleteLater()
        self.fitworker = None

    def onDataChanged(self):
        self.onNewFitCurve()
        for i in range(self.model.columnCount()):
            self.treeView.resizeColumnToContents(i)

    def onSelectionChanged(self):
        any_selected = self.treeView.currentIndex().isValid()
        self.m5ToolButton.setEnabled(any_selected and self.dataset is not None)
        self.p5ToolButton.setEnabled(any_selected and self.dataset is not None)
        self.m20ToolButton.setEnabled(any_selected and self.dataset is not None)
        self.p20ToolButton.setEnabled(any_selected and self.dataset is not None)

    def adjustSelectedParameter(self, percentile):
        idx = self.treeView.currentIndex()
        if not idx.isValid():
            return
        currvalue = self.model.parameter(idx.row()).value
        self.model.changeValue(idx.row(), currvalue*percentile)
        self.onNewFitCurve()

    def onMinus5(self):
        self.adjustSelectedParameter(0.95)

    def onPlus5(self):
        self.adjustSelectedParameter(1.05)

    def onMinus20(self):
        self.adjustSelectedParameter(0.80)

    def onPlus20(self):
        self.adjustSelectedParameter(1.20)

    def setModelFunc(self, modelfunc:fffs.ModelFunction):
        self.func = modelfunc
        newmodel = ParameterModel()
        self.treeView.selectionModel().currentRowChanged.disconnect()
        self.treeView.setModel(newmodel)
        self.treeView.selectionModel().currentRowChanged.connect(self.onSelectionChanged)
        self.model.dataChanged.disconnect()
        self.model.historyChanged.disconnect()
        self.model.cleanup()
        self.model = newmodel
        self.model.dataChanged.connect(self.onDataChanged)
        self.model.historyChanged.connect(self.onHistoryChanged)
        self.onHistoryChanged()
        self.historyHorizontalSlider.valueChanged.connect(self.onHistoryMoveRequest)
        self.historyBackToolButton.clicked.connect(self.onHistoryBack)
        self.historyForwardToolButton.clicked.connect(self.onHistoryForward)
        for p in self.func.parameters:
            self.model.addParameter(p.name, p.defaultvalue, p.lbound, p.ubound, p.fittable)
        self.onSelectionChanged()
        self.onNewFitCurve()

    def onHistoryBack(self):
        self.model.historyBack()

    def onHistoryForward(self):
        self.model.historyForward()

    def onHistoryMoveRequest(self):
        self.model.historyGoto(int(self.historyHorizontalSlider.value()))

    def onResetParameters(self):
        self.setModelFunc(self.func)

    def onClearHistory(self):
        self.model.historyReset()

    def onHistoryChanged(self):
        self.historyHorizontalSlider.setMinimum(0)
        self.historyHorizontalSlider.setMaximum(self.model.historySize()-1)
        try:
            self.historyHorizontalSlider.blockSignals(True)
            self.historyHorizontalSlider.setValue(self.model.historyIndex())
        finally:
            self.historyHorizontalSlider.blockSignals(False)

    def setDataSet(self, dataset:np.ndarray):
        self.dataset = dataset
        self.fitcurve = None
        self.onSelectionChanged()
        self.onNewFitCurve()

    def onNewFitCurve(self):
        if self.dataset is None:
            return
        if self.func is None:
            return
        if self.model is None:
            return
        parameters = [self.model.parameter(i).value for i in range(self.model.rowCount())]
        self.fitcurve = self.func.fitfunction(self.dataset[:,0], *parameters)
        self.fitCurveChanged.emit(self.fitCurve())

    def fitCurve(self):
        return np.stack((self.dataset[:,0],self.fitcurve)).T

