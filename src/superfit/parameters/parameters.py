from PyQt5 import QtWidgets, QtCore
import fffs
import time
import scipy.optimize
import traceback
from scipy.linalg import svd
from .parametermodel import ParameterModel, Parameter
from .parameters_ui import Ui_Form
import sys
import numpy as np
from typing import Sequence, Callable, Type, List, Dict

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
    R2_w:float=None
    R2_adj:float=None
    R2_adj_w:float=None
    chi2:float=None
    chi2_reduced:float=None
    dof:int=None
    elapsedtime:float=None
    nfunceval:float=None
    message:str=None
    statusint:int=None
    sstot:float=None
    ssres:float=None
    sstot_w:float=None
    ssres_w:float=None


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
            self.dy = np.ones_like(self.x)
        if kwargs.pop('dy_weight'):
            self.dy = np.ones_like(self.x)
        try:
            self.dx = dataset[:,3]
        except IndexError:
            self.dx = np.ones_like(self.x)
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
    fitResultsReady = QtCore.pyqtSignal(FitResults)
    dataset: np.ndarray = None
    fitworker: FitWorker = None
    algorithm_kwargs: Dict = {}
    fitstarted:float = None
    lastresults: FitResults=None

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
        self.slackenLimitsPushButton.clicked.connect(self.onFixOOBLimits)
        self.correctValuesPushButton.clicked.connect(self.onFixOOBParameters)
        self.checkOutOfBoundsParameters()
        self.onSelectionChanged()

    def onStep(self):
        max_nfev_orig = self.algorithm_kwargs['max_nfev']
        npars = len([i for i in range(self.model.rowCount())
                     if self.model.parameter(i).fittable and not self.model.parameter(i).fixed])
        try:
            if self.algorithm_kwargs['method'] in ['trf', 'dogbox']:
                self.algorithm_kwargs['max_nfev'] = npars
            elif self.algorithm_kwargs['method'] == 'lm':
                self.algorithm_kwargs['max_nfev'] = npars*(npars+1)
            else:
                raise NotImplementedError(self.algorithm_kwargs['method'])
            self.onFit()
        finally:
            self.algorithm_kwargs['max_nfev'] = max_nfev_orig

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
             for i in range(self.model.rowCount())], **self.algorithm_kwargs
        )
        self.fitworker.resultsReady.connect(self.onFitResultsReady)
        self.fitworker.exceptionOccurred.connect(self.onFitException)
        self.fitworker.finished.connect(self.onFitFinished)
        self.fitworker.started.connect(self.onFitStarted)
        self.fitworker.start()

    def onFitStarted(self):
        self.checkOutOfBoundsParameters()
#        self.fitPushButton.setEnabled(False)
#        self.stepPushButton.setEnabled(False)
        self.fitstarted = time.monotonic()

    def onFitResultsReady(self, results:scipy.optimize.OptimizeResult):
        fitresults = FitResults()
        fitresults.elapsedtime = time.monotonic()- self.fitstarted
        fitresults.dataset = self.dataset
        fitresults.paramnames = [self.model.parameter(i).name for i in range(self.model.rowCount())]
        self.fitstarted = None
        fitresults.message = results.message
        fitresults.nfunceval = results.nfev
        fitresults.paramlboundactive = [self.model.parameter(i).lbound_enabled for i in range(self.model.rowCount())]
        fitresults.paramlbounds = [self.model.parameter(i).lbound for i in range(self.model.rowCount())]
        fitresults.paramuboundactive= [self.model.parameter(i).ubound_enabled for i in range(self.model.rowCount())]
        fitresults.paramubounds = [self.model.parameter(i).ubound for i in range(self.model.rowCount())]
        fitresults.statusint = results.status
        fitresults.paramvalues = np.array([self.model.parameter(i).value for i in range(self.model.rowCount())])
        fitresults.paramfree = np.array([self.model.parameter(i).fittable and not self.model.parameter(i).fixed
                         for i in range(self.model.rowCount())])
        fitresults.paramvalues[fitresults.paramfree] = results.x
        fitresults.funcvalue = self.func.fitfunction(self.dataset[:,0], *fitresults.paramvalues)
        _, s, VT = svd(results.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(results.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        fitresults.covar = np.dot(VT.T / s ** 2, VT)
        fitresults.paramuncertainties = np.zeros_like(fitresults.paramvalues)
        fitresults.paramuncertainties[fitresults.paramfree] = np.diag(fitresults.covar)**0.5
        fitresults.paramactive_mask = np.zeros(len(fitresults.paramvalues), dtype=np.uint8)
        fitresults.paramactive_mask[fitresults.paramfree] = results.active_mask
        fitresults.correl_coeffs = (fitresults.covar / np.outer(np.diag(fitresults.covar)**0.5, np.diag(fitresults.covar)**0.5))
        fitresults.chi2 = (results.fun**2).sum()
        fitresults.dof = self.dataset.shape[0]-fitresults.paramfree.sum()
        fitresults.chi2_reduced = fitresults.chi2/fitresults.dof
        fitresults.sstot = ((self.dataset[:,1]-np.mean(self.dataset[:,1]))**2).sum()
        fitresults.ssres = ((fitresults.funcvalue - self.dataset[:,1])**2).sum()
        fitresults.R2 = 1-fitresults.ssres/fitresults.sstot
        fitresults.R2_adj = 1 - (fitresults.ssres/ (fitresults.dof-1)) / (fitresults.sstot/(self.dataset.shape[0]-1))
        if self.dataset.shape[1]>2:
            fitresults.sstot_w = (((self.dataset[:, 1] - np.mean(self.dataset[:, 1]))/self.dataset[:,2]) ** 2).sum()
            fitresults.ssres_w = (((fitresults.funcvalue - self.dataset[:, 1])/self.dataset[:,2]) ** 2).sum()
            fitresults.R2_w = 1-fitresults.ssres_w/fitresults.sstot_w
            fitresults.R2_adj_w = 1- (fitresults.ssres_w / (fitresults.dof-1)) / (fitresults.sstot_w/(self.dataset.shape[0]-1))
        else:
            fitresults.sstot_w = fitresults.sstot
            fitresults.ssres_w = fitresults.ssres
            fitresults.R2_w = fitresults.R2
            fitresults.R2_adj_w = fitresults.R2_adj

        self.lastresults = fitresults
        self.updateWithResults(fitresults)
        self.fitResultsReady.emit(fitresults)

    def updateWithResults(self, fitresults:FitResults):
        for i in range(self.model.rowCount()):
            if fitresults.paramfree[i]:
                self.model.changeValue(i, fitresults.paramvalues[i])
                self.model.changeUncertainty(i, fitresults.paramuncertainties[i])
            else:
                self.model.changeUncertainty(i, 0)
            self.model.changeLBoundActive(i, fitresults.paramactive_mask[i]<0)
            self.model.changeUBoundActive(i, fitresults.paramactive_mask[i]>0)
        self.exitStatusLabel.setText('{} (code: {:d})'.format(fitresults.message, fitresults.statusint))
        self.chi2Label.setText('{:.4f}'.format(fitresults.chi2))
        self.reducedChi2Label.setText('{:.4f}'.format(fitresults.chi2_reduced))
        self.r2Label.setText('{:.4f}'.format(fitresults.R2_adj))
        self.dofLabel.setText('{}'.format(fitresults.dof))
        self.elapsedTimeLabel.setText('{:.4f} secs'.format(fitresults.elapsedtime))
        self.nfuncevLabel.setText('{}'.format(fitresults.nfunceval))
        self.model.historyPush()
        self.onNewFitCurve()

    def onFitException(self, exctype:Type[Exception], exc:Exception, tb:{'tb_frame', 'tb_lineno'}):
        QtWidgets.QMessageBox.critical(self, 'Error while fitting', '\n'.join(traceback.format_exception(exctype, exc, tb)))

    def onFitFinished(self):
        self.fitworker.deleteLater()
        self.fitworker = None
        self.checkOutOfBoundsParameters()
        #self.fitPushButton.setEnabled(True)
        #self.stepPushButton.setEnabled(True)

    def onDataChanged(self):
        self.onNewFitCurve()
        for i in range(self.model.columnCount()):
            self.treeView.resizeColumnToContents(i)
        self.checkOutOfBoundsParameters()

    def checkOutOfBoundsParameters(self):
        oob = [p.name for p in self.model if ((p.value < p.lbound) and p.lbound_enabled) or
               ((p.value > p.ubound) and p.ubound_enabled) ]
        self.oobLabel.setText('The following parameters are out of bounds: {}'.format(', '.join(oob)))
        self.outOfBoundParametersGroupBox.setVisible(bool(oob))
        self.fitPushButton.setEnabled((self.dataset is not None) and (not bool(oob)) and (self.fitworker is None))
        self.stepPushButton.setEnabled((self.dataset is not None) and (not bool(oob)) and (self.fitworker is None))

    def onFixOOBLimits(self):
        for p in self.model:
            if (p.value< p.lbound) and (p.lbound_enabled):
                self.model.changeLBound(p.name, p.value)
            if (p.value > p.ubound) and (p.ubound_enabled):
                self.model.changeUBound(p.name, p.value)
        self.checkOutOfBoundsParameters()

    def onFixOOBParameters(self):
        for p in self.model:
            if (p.value< p.lbound) and (p.lbound_enabled):
                self.model.changeValue(p.name, p.lbound)
            if (p.value > p.ubound) and (p.ubound_enabled):
                self.model.changeValue(p.name, p.ubound)
        self.checkOutOfBoundsParameters()

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
        for p in self.func.parameters:
            self.model.addParameter(p.name, p.defaultvalue, p.lbound, p.ubound, p.fittable)
        self.model.dataChanged.connect(self.onDataChanged)
        self.model.historyChanged.connect(self.onHistoryChanged)
        self.historyHorizontalSlider.valueChanged.connect(self.onHistoryMoveRequest)
        self.historyBackToolButton.clicked.connect(self.onHistoryBack)
        self.historyForwardToolButton.clicked.connect(self.onHistoryForward)
        for i in range(self.model.columnCount()):
            self.treeView.resizeColumnToContents(i)
        self.onSelectionChanged()
        self.onHistoryChanged()
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
        self.onNewFitCurve()

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

    def setAlgorithmKwargs(self, kwargs):
        self.algorithm_kwargs = kwargs