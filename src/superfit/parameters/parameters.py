from PyQt5 import QtWidgets, QtCore
import fffs
import datetime
import time
import queue
import scipy.optimize
import multiprocessing
import traceback
from matplotlib.figure import Figure
from scipy.linalg import svd
from .parametermodel import ParameterModel, Parameter, SpinBoxDelegate
from .parameters_ui import Ui_Form
import sys
import numpy as np
from typing import Sequence, Callable, Type, List, Dict


class FitResults:
    dataset: np.ndarray = None
    funcvalue: np.ndarray = None
    paramnames: List[str] = None
    paramvalues: np.ndarray = None
    paramuncertainties: np.ndarray = None
    paramfree: np.ndarray = None
    paramlbounds: np.ndarray = None
    paramubounds: np.ndarray = None
    paramlboundactive: np.ndarray = None
    paramuboundactive: np.ndarray = None
    paramactive_mask: np.ndarray = None
    covar: np.ndarray = None
    correl_coeffs: np.ndarray = None
    R2: float = None
    R2_w: float = None
    R2_adj: float = None
    R2_adj_w: float = None
    chi2: float = None
    chi2_reduced: float = None
    dof: int = None
    elapsedtime: float = None
    nfunceval: float = None
    message: str = None
    statusint: int = None
    sstot: float = None
    ssres: float = None
    sstot_w: float = None
    ssres_w: float = None


def do_fitting(resultsqueue: multiprocessing.Queue,
               function: Callable, dataset: np.ndarray, param_init: Sequence,
               fittable: Sequence, lbounds: Sequence, ubounds: Sequence, **kwargs):
    try:
        free = np.array(fittable, dtype=np.bool)
        param_init = np.array(param_init, dtype=np.double)
        x = dataset[:, 0]
        y = dataset[:, 1]
        try:
            dy = dataset[:, 2]
        except IndexError:
            dy = np.ones_like(x)
        if kwargs.pop('dy_weight'):
            dy = np.ones_like(x)
        try:
            dx = dataset[:, 3]
        except IndexError:
            dx = np.ones_like(x)

        lbounds = np.array(lbounds)
        ubounds = np.array(ubounds)
        lsq_kwargs = kwargs

        def fitfunc(params: np.ndarray, param_init, free, x, y, dy, function):
            param_init[free] = params
            return (y - function(x, *param_init)) / dy

        resultsqueue.put_nowait(('started', None))
        result = scipy.optimize.least_squares(
            fitfunc, param_init[free],
            bounds=(lbounds[free], ubounds[free]),
            args=(param_init, free, x, y, dy, function),
            **lsq_kwargs
        )
        resultsqueue.put_nowait(('result', result))
    except Exception:
        resultsqueue.put_nowait(('exception', *sys.exc_info()))


class ParameterView(QtWidgets.QWidget, Ui_Form):
    func: fffs.ModelFunction = None
    fitcurve: np.ndarray = None
    fitCurveChanged = QtCore.pyqtSignal(np.ndarray)
    fitResultsReady = QtCore.pyqtSignal(FitResults)
    exportReportRequested = QtCore.pyqtSignal()
    dataset: np.ndarray = None
    algorithm_kwargs: Dict = {}
    fitstarted: float = None
    lastresults: FitResults = None
    fitworker: multiprocessing.Process = None
    resultsqueue: multiprocessing.Queue = None
    resultstimer: int

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.model = ParameterModel()
        self.model.dataChanged.connect(self.onDataChanged)
        self.model.historyChanged.connect(self.onHistoryChanged)
        self.delegate = SpinBoxDelegate()
        self.treeView.setModel(self.model)
        self.treeView.setItemDelegateForColumn(1, self.delegate)
        self.treeView.setItemDelegateForColumn(2, self.delegate)
        self.treeView.setItemDelegateForColumn(3, self.delegate)
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
        self.stopFittingPushButton.setVisible(False)
        self.progressBar.setVisible(False)
        self.stopFittingPushButton.clicked.connect(self.onStopFitting)
        self.savePushButton.clicked.connect(self.onSaveParameters)
        self.loadPushButton.clicked.connect(self.onLoadParameters)
        self.exportReportPushButton.clicked.connect(self.exportReportRequested.emit)

    def onSaveParameters(self):
        filename, filtername = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save fit parameters to...", '', "Superfit state files (*.ssf);; All files (*)",
            "Superfit state files (*.ssf)")
        if not filename:
            return
        if not filename.endswith('.ssf'):
            filename = filename + '.ssf'
        self.model.saveState(filename)

    def onLoadParameters(self):
        filename, filtername = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load fit parameters from...", '', 'Superfit state files (*.ssf);; All files (*)',
            "Superfit state files (*.ssf)"
        )
        if not filename:
            return
        try:
            self.model.loadState(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Error while loading state file', str(exc))

    def onStep(self):
        max_nfev_orig = self.algorithm_kwargs['max_nfev']
        npars = len([i for i in range(self.model.rowCount())
                     if self.model.parameter(i).fittable and not self.model.parameter(i).fixed])
        try:
            if self.algorithm_kwargs['method'] in ['trf', 'dogbox']:
                self.algorithm_kwargs['max_nfev'] = npars
            elif self.algorithm_kwargs['method'] == 'lm':
                self.algorithm_kwargs['max_nfev'] = npars * (npars + 1)
            else:
                raise NotImplementedError(self.algorithm_kwargs['method'])
            self.onFit()
        finally:
            self.algorithm_kwargs['max_nfev'] = max_nfev_orig

    def onFit(self):
        if self.fitworker is not None:
            return
        self.resultsqueue = multiprocessing.Queue()
        self.fitworker = multiprocessing.Process(target=do_fitting, args=(
            self.resultsqueue, self.func.fitfunction, self.dataset,
            [self.model.parameter(i).value for i in range(self.model.rowCount())],
            [self.model.parameter(i).fittable and not self.model.parameter(i).fixed
             for i in range(self.model.rowCount())],
            [(-np.inf, self.model.parameter(i).lbound)[self.model.parameter(i).lbound_enabled]
             for i in range(self.model.rowCount())],
            [(np.inf, self.model.parameter(i).ubound)[self.model.parameter(i).ubound_enabled]
             for i in range(self.model.rowCount())],
        ), kwargs=self.algorithm_kwargs)
        self.fitworker.daemon = True
        self.resultstimer = self.startTimer(100)
        self.fitworker.start()

    def timerEvent(self, event: QtCore.QTimerEvent):
        if event.timerId() != self.resultstimer:
            super().timerEvent(event)
            return
        event.accept()
        try:
            what, obj = self.resultsqueue.get_nowait()
        except queue.Empty:
            return
        if what == 'started':
            self.onFitStarted()
        elif what == 'result':
            self.onFitResultsReady(obj)
            self.finalizeFitWorker()
        elif what == 'exception':
            self.onFitException(*obj)
            self.finalizeFitWorker()

    def finalizeFitWorker(self):
        self.fitworker.join()
        self.fitworker = None
        self.resultsqueue.close()
        self.resultsqueue = None
        self.killTimer(self.resultstimer)
        self.resultstimer = None
        self.onFitFinished()

    def onStopFitting(self):
        if self.fitworker is None:
            return
        self.fitworker.terminate()

    def onFitStarted(self):
        self.checkOutOfBoundsParameters()
        #        self.fitPushButton.setEnabled(False)
        #        self.stepPushButton.setEnabled(False)
        self.fitstarted = time.monotonic()
        self.progressBar.setVisible(True)
        self.stopFittingPushButton.setVisible(True)

    def onFitResultsReady(self, results: scipy.optimize.OptimizeResult):
        fitresults = FitResults()
        fitresults.elapsedtime = time.monotonic() - self.fitstarted
        fitresults.dataset = self.dataset
        fitresults.paramnames = [self.model.parameter(i).name for i in range(self.model.rowCount())]
        self.fitstarted = None
        fitresults.message = results.message
        fitresults.nfunceval = results.nfev
        fitresults.paramlboundactive = [self.model.parameter(i).lbound_enabled for i in range(self.model.rowCount())]
        fitresults.paramlbounds = [self.model.parameter(i).lbound for i in range(self.model.rowCount())]
        fitresults.paramuboundactive = [self.model.parameter(i).ubound_enabled for i in range(self.model.rowCount())]
        fitresults.paramubounds = [self.model.parameter(i).ubound for i in range(self.model.rowCount())]
        fitresults.statusint = results.status
        fitresults.paramvalues = np.array([self.model.parameter(i).value for i in range(self.model.rowCount())])
        fitresults.paramfree = np.array([self.model.parameter(i).fittable and not self.model.parameter(i).fixed
                                         for i in range(self.model.rowCount())])
        fitresults.paramvalues[fitresults.paramfree] = results.x
        fitresults.funcvalue = self.func.fitfunction(self.dataset[:, 0], *fitresults.paramvalues)
        _, s, VT = svd(results.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(results.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        fitresults.covar = np.dot(VT.T / s ** 2, VT)
        fitresults.paramuncertainties = np.zeros_like(fitresults.paramvalues)
        fitresults.paramuncertainties[fitresults.paramfree] = np.diag(fitresults.covar) ** 0.5
        fitresults.paramactive_mask = np.zeros(len(fitresults.paramvalues), dtype=np.int8)
        fitresults.paramactive_mask[fitresults.paramfree] = results.active_mask
        fitresults.correl_coeffs = (
                    fitresults.covar / np.outer(np.diag(fitresults.covar) ** 0.5, np.diag(fitresults.covar) ** 0.5))
        fitresults.chi2 = (results.fun ** 2).sum()
        fitresults.dof = self.dataset.shape[0] - fitresults.paramfree.sum()
        fitresults.chi2_reduced = fitresults.chi2 / fitresults.dof
        fitresults.sstot = ((self.dataset[:, 1] - np.mean(self.dataset[:, 1])) ** 2).sum()
        fitresults.ssres = ((fitresults.funcvalue - self.dataset[:, 1]) ** 2).sum()
        fitresults.R2 = 1 - fitresults.ssres / fitresults.sstot
        fitresults.R2_adj = 1 - (fitresults.ssres / (fitresults.dof - 1)) / (
                    fitresults.sstot / (self.dataset.shape[0] - 1))
        if self.dataset.shape[1] > 2:
            fitresults.sstot_w = (((self.dataset[:, 1] - np.mean(self.dataset[:, 1])) / self.dataset[:, 2]) ** 2).sum()
            fitresults.ssres_w = (((fitresults.funcvalue - self.dataset[:, 1]) / self.dataset[:, 2]) ** 2).sum()
            fitresults.R2_w = 1 - fitresults.ssres_w / fitresults.sstot_w
            fitresults.R2_adj_w = 1 - (fitresults.ssres_w / (fitresults.dof - 1)) / (
                        fitresults.sstot_w / (self.dataset.shape[0] - 1))
        else:
            fitresults.sstot_w = fitresults.sstot
            fitresults.ssres_w = fitresults.ssres
            fitresults.R2_w = fitresults.R2
            fitresults.R2_adj_w = fitresults.R2_adj

        self.lastresults = fitresults
        self.updateWithResults(fitresults)
        self.fitResultsReady.emit(fitresults)

    def updateWithResults(self, fitresults: FitResults):
        try:
            self.model.blockSignals(True)
            for i in range(self.model.rowCount()):
                if fitresults.paramfree[i]:
                    self.model.changeValue(i, fitresults.paramvalues[i])
                    self.model.changeUncertainty(i, fitresults.paramuncertainties[i])
                else:
                    self.model.changeUncertainty(i, 0)
                self.model.changeLBoundActive(i, fitresults.paramactive_mask[i] < 0)
                self.model.changeUBoundActive(i, fitresults.paramactive_mask[i] > 0)
            self.exitStatusLabel.setText('{} (code: {:d})'.format(fitresults.message, fitresults.statusint))
            self.chi2Label.setText('{:.4f}'.format(fitresults.chi2))
            self.reducedChi2Label.setText('{:.4f}'.format(fitresults.chi2_reduced))
            self.r2Label.setText('{:.4f}'.format(fitresults.R2_adj))
            self.dofLabel.setText('{}'.format(fitresults.dof))
            self.elapsedTimeLabel.setText('{:.4f} secs'.format(fitresults.elapsedtime))
            self.nfuncevLabel.setText('{}'.format(fitresults.nfunceval))
        finally:
            self.model.blockSignals(False)
        self.model.historyPush()
        self.onNewFitCurve()

    def onFitException(self, exctype: Type[Exception], exc: Exception, tb: {'tb_frame', 'tb_lineno'}):
        QtWidgets.QMessageBox.critical(self, 'Error while fitting',
                                       '\n'.join(traceback.format_exception(exctype, exc, tb)))

    def onFitFinished(self):
        self.checkOutOfBoundsParameters()
        self.progressBar.setVisible(False)
        self.stopFittingPushButton.setVisible(False)
        # self.fitPushButton.setEnabled(True)
        # self.stepPushButton.setEnabled(True)

    def onDataChanged(self, topleft: QtCore.QModelIndex, bottomright: QtCore.QModelIndex, roles: List[int]):
        if topleft.column() <= 3 and bottomright.column() >= 3 and QtCore.Qt.DisplayRole in roles:
            # only do work if a parameter value has changed
            self.onNewFitCurve()
            for i in range(self.model.columnCount()):
                self.treeView.resizeColumnToContents(i)
            self.checkOutOfBoundsParameters()

    def checkOutOfBoundsParameters(self):
        oob = [p.name for p in self.model if ((p.value < p.lbound) and p.lbound_enabled) or
               ((p.value > p.ubound) and p.ubound_enabled)]
        self.oobLabel.setText('The following parameters are out of bounds: {}'.format(', '.join(oob)))
        self.outOfBoundParametersGroupBox.setVisible(bool(oob))
        self.fitPushButton.setEnabled((self.dataset is not None) and (not bool(oob)) and (self.fitworker is None))
        self.stepPushButton.setEnabled((self.dataset is not None) and (not bool(oob)) and (self.fitworker is None))

    def onFixOOBLimits(self):
        for p in self.model:
            if (p.value < p.lbound) and (p.lbound_enabled):
                self.model.changeLBound(p.name, p.value)
            if (p.value > p.ubound) and (p.ubound_enabled):
                self.model.changeUBound(p.name, p.value)
        self.checkOutOfBoundsParameters()

    def onFixOOBParameters(self):
        for p in self.model:
            if (p.value < p.lbound) and (p.lbound_enabled):
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
        self.model.changeValue(idx.row(), currvalue * percentile)
        #self.onNewFitCurve()

    def onMinus5(self):
        self.adjustSelectedParameter(0.95)

    def onPlus5(self):
        self.adjustSelectedParameter(1.05)

    def onMinus20(self):
        self.adjustSelectedParameter(0.80)

    def onPlus20(self):
        self.adjustSelectedParameter(1.20)

    def setModelFunc(self, modelfunc: fffs.ModelFunction):
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
            self.model.addParameter(p.name, p.defaultvalue, p.lbound, p.ubound, p.fittable,
                                    lbound_enabled=np.isfinite(p.lbound),
                                    ubound_enabled=np.isfinite(p.ubound),
                                    description=p.description)
        self.model.dataChanged.connect(self.onDataChanged)
        self.model.historyChanged.connect(self.onHistoryChanged)
        self.historyHorizontalSlider.valueChanged.connect(self.onHistoryMoveRequest)
        self.historyBackToolButton.clicked.connect(self.onHistoryBack)
        self.historyForwardToolButton.clicked.connect(self.onHistoryForward)
        self.model.historyPush()
        for i in range(self.model.columnCount()):
            self.treeView.resizeColumnToContents(i)
        self.onSelectionChanged()
        self.onHistoryChanged()
        self.checkOutOfBoundsParameters()

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
        self.historyHorizontalSlider.setMaximum(self.model.historySize() - 1)
        try:
            self.historyHorizontalSlider.blockSignals(True)
            self.historyHorizontalSlider.setValue(self.model.historyIndex())
        finally:
            self.historyHorizontalSlider.blockSignals(False)
        self.onNewFitCurve()

    def setDataSet(self, dataset: np.ndarray):
        self.dataset = dataset
        self.fitcurve = None
        self.onSelectionChanged()
        self.onHistoryChanged()
        self.checkOutOfBoundsParameters()

    def onNewFitCurve(self):
        if self.dataset is None:
            return
        if self.func is None:
            return
        if self.model is None:
            return
        parameters = [self.model.parameter(i).value for i in range(self.model.rowCount())]
        self.fitcurve = self.func.fitfunction(self.dataset[:, 0], *parameters)
        self.fitCurveChanged.emit(self.fitCurve())

    def fitCurve(self):
        return np.stack((self.dataset[:, 0], self.fitcurve)).T

    def setAlgorithmKwargs(self, kwargs):
        self.algorithm_kwargs = kwargs

    def plotRepresentation(self, fig: Figure):
        fig.clf()
        self.func.visualize(fig, self.dataset[:, 0], *[p.value for p in self.model])
        fig.canvas.draw()

