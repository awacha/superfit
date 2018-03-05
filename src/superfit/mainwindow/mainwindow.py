from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import fffs
import os
import datetime
import shutil
from docutils.core import publish_cmdline_to_binary, default_description
from docutils.writers.odf_odt import Reader, Writer
from typing import Dict
from .mainwindow_ui import Ui_MainWindow
from ..datasetselector import DataSetSelector
from ..parameters import ParameterView, FitResults
from ..modelselector import ModelSelector
from ..lsqalgorithmselector import LSQAlgorithmSelector
from ..correlation import CorrelationCoefficientsTable
from ..modelrepresentation import ModelRepresentation
from ..graph import Graph
import pkginfo
import tempfile

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _data: np.ndarray=None
    _model: fffs.ModelFunction=None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pagewidgets = []
        self.setupUi(self)
        self._data = None

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.toolBox.removeItem(0)
        del self.page
        self.setWindowTitle('SuperFit v{}'.format(pkginfo.get_metadata('superfit').version))
        self.dataselector = DataSetSelector(self)
        self.modelselector = ModelSelector(self)
        self.parameterview = ParameterView(self)
        self.lsqalgorithmselector = LSQAlgorithmSelector(self)
        self.correlationcoefficients = CorrelationCoefficientsTable(self)
        self.modelrepresentation = ModelRepresentation(self)
        for text, attrname in [
            ('Load data...', 'dataselector'),
            ('Model function', 'modelselector'),
            ('Algorithm tuning', 'lsqalgorithmselector'),
            ('Parameters && fitting...', 'parameterview')]:
            w=QtWidgets.QWidget(self.toolBox)
            w.setContentsMargins(0,0,0,0)
            w.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
            l=QtWidgets.QVBoxLayout(w)
            l.setContentsMargins(0,0,0,0)
            w.setLayout(l)
            l.addWidget(getattr(self, attrname))
            #l.addStretch(0)
            self._pagewidgets.append(w)
            self.toolBox.addItem(self._pagewidgets[-1], text)
        self.dataselector.dataSetLoaded.connect(self.onDataLoaded)
        self.modelselector.modelSelected.connect(self.onModelSelected)
        self.parameterview.fitCurveChanged.connect(self.onFitCurveChanged)
        self.parameterview.fitResultsReady.connect(self.onFitResultsReady)
        self.lsqalgorithmselector.algorithmChanged.connect(self.onAlgorithmChanged)
        self.tabWidget.removeTab(0)
        self.graph = Graph(self)
        self.tabWidget.addTab(self.graph,'Data && fit')
        self.tabWidget.addTab(self.modelrepresentation, 'Model representation')
        self.tabWidget.addTab(self.correlationcoefficients, 'Parameter correlation')
#        self.tabWidget.addTab(..., 'Result && statistics')
        self.onModelSelected(self.modelselector.model())
        self.onAlgorithmChanged(self.lsqalgorithmselector.algorithmKwargs())
        self.parameterview.exportReportRequested.connect(self.onExportResults)

    def onFitResultsReady(self, fitresults:FitResults):
        self.correlationcoefficients.setMatrix(
            fitresults.correl_coeffs,
            [p for p, f in zip(fitresults.paramnames, fitresults.paramfree) if f])

    def onDataLoaded(self, data:np.ndarray):
        self._data = data
        self.graph.setDataSet(self._data)
        self.parameterview.setDataSet(self._data)
        self.graph.replotData()

    def onModelSelected(self, model:fffs.ModelFunction):
        self._model = model
        self.parameterview.setModelFunc(self._model)

    def onAlgorithmChanged(self, kwargs:Dict):
        self.parameterview.setAlgorithmKwargs(kwargs)

    def plotData(self):
        self.graph.replotData()

    def plotFitted(self):
        self.graph.replotModel()

    def onFitCurveChanged(self, fitcurve:np.ndarray):
        self.graph.setFitCurve(fitcurve)
        self.graph.replotModel()
        self.parameterview.plotRepresentation(self.modelrepresentation.figure)

    def onExportResults(self):
        filename, filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Write report to file...", '',
            'ODF Text Document (*.odt);; All files (*)', 'ODF Text Document (*.odt)')
        if not filename:
            return
        if not filename.endswith('.odt'):
            filename = filename+'.odt'
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, 'report.rst'), 'wt') as f:
                f.write('===============\nFitting results\n===============\n\n*Date*: {}\n\n'.format(datetime.datetime.now()))
                f.write('*superfit version:* {}\n\n'.format(pkginfo.get_metadata('superfit').version))
                f.write('*fffs version:* {}\n\n'.format(pkginfo.get_metadata('fffs').version))
                f.write('\nGraph\n=====\n')
                f.write('.. figure:: report.png\n'.format(os.path.split(filename)[-1].rsplit('.',1)[0]))
                f.write('   :alt: fit curve\n\n')
                f.write('   The measured and the fitted curve\n')
                f.write('\nDataset\n=======\n')
                f.write('*File name:* {}\n\n'.format(self.dataselector.fileNameLineEdit.text()))
                ds = self.dataselector.dataSet()
                f.write('*Minimum x:* {}\n\n'.format(ds[:,0].min()))
                f.write('*Maximum x:* {}\n\n'.format(ds[:,0].max()))
                f.write('*Number of points:* {}\n\n'.format(ds.shape[0]))
                f.write('\nModel\n=====\n')
                modelfunc=self.modelselector.model()
                f.write('*Name:* {}\n\n'.format(modelfunc.name))
                f.write('*Description:* {}\n\n'.format(modelfunc.description))
                f.write('*Category:* {}\n\n'.format(modelfunc.category))
                f.write('*Subcategory:* {}\n\n'.format(modelfunc.subcategory))
                algorithm=self.lsqalgorithmselector.algorithmKwargs()
                f.write('\nAlgorithm\n=========\n')
                for k in algorithm:
                    f.write('*{}:* {}\n'.format(k, algorithm[k]))
                f.write('\nResults\n=======\n')
                f.write('\nStatistics\n----------\n')
                r = self.parameterview.lastresults
                for attr, label in [
                    ('R2', 'Coefficient of determination (R^2)'),
                    ('R2_w', 'Weighted coefficient of determination (R^2)'),
                    ('R2_adj', 'Adjusted coefficient of determination (R^2'),
                    ('R2_adj_w', 'Adjusted, weighted coefficient of determination (R^2)'),
                    ('chi2', 'Chi2'),
                    ('chi2_reduced', 'Reduced chi^2'),
                    ('dof', 'Degrees of freedom (DoF)'),
                    ('elapsedtime', 'Elapsed time (sec)'),
                    ('nfunceval', 'Number of function evaluations'),
                    ('statusint', 'Algorithm exit status'),
                    ('message', 'Algorithm exit reason'),
                    ('sstot', 'SS_tot'),
                    ('ssres', 'SS_res'),
                    ('sstot_w', 'Weighted SS_tot'),
                    ('ssres_w', 'Weighted SS_res'),
                ]:
                    f.write('*{}:* {}\n\n'.format(label, getattr(r, attr)))
                f.write('\nFitted parameter values\n-----------------------\n')
                for name, value, uncertainty, free, lbound, ubound, lboundactive, uboundactive, activemask in zip(
                    r.paramnames, r.paramvalues, r.paramuncertainties, r.paramfree, r.paramlbounds,
                    r.paramubounds, r.paramlboundactive, r.paramuboundactive, r.paramactive_mask):
                    if not lboundactive:
                        lbound='unlimited'
                    else:
                        lbound='{:.6g}'.format(lbound)
                    if not uboundactive:
                        ubound='unlimited'
                    else:
                        ubound='{:.6g}'.format(ubound)
                    if not free:
                        uncertainty='(fixed)'
                        pcnt=''
                    else:
                        if value==0:
                            pcnt = ' (NaN %)'
                        else:
                            pcnt = ' ({:.2f} %)'.format(uncertainty/abs(value)*100)
                        uncertainty='\xb1 {:.8g}'.format(uncertainty)+pcnt

                    lboundhit = ['','(!!!)'][activemask<0]
                    uboundhit = ['','(!!!)'][activemask>0]
                    f.write('* **{}** ({}{}..{}{}): {:.8g} {}\n\n'.format(
                        name, lbound, lboundhit, ubound, uboundhit, value, uncertainty))
                f.write('\nParameter correlation\n---------------------\n')
                freenames = [n for n,f in zip(r.paramnames,r.paramfree) if f]
                maxnamelen = max(max([len(n) for n in freenames]),2)
                f.write('+'+('-'*(maxnamelen+6)+'+')*(len(freenames)+1)+'\n')
                f.write('|'+' '*(maxnamelen+6)+'|')
                for n in freenames:
                    f.write(' {{:^{}s}} |'.format(maxnamelen+4).format('**'+n+'**'))
                f.write('\n')
                f.write('+'+('-'*(maxnamelen+6)+'+')*(len(freenames)+1)+'\n')
                for i, n in enumerate(freenames):
                    f.write('| {{:<{}s}} |'.format(maxnamelen+4).format('**'+n+'**'))
                    f.write(''.join([' {{:^+{}.3f}} |'.format(maxnamelen+4).format(x) for x in r.correl_coeffs[i, :]])+'\n')
                    f.write('+' + ('-' * (maxnamelen + 6) + '+') * (len(freenames) + 1) + '\n')
            self.graph.figure.savefig(os.path.join(td,'report.png'),dpi=300)
        description = ('Generates OpenDocument/OpenOffice/ODF documents from '
               'standalone reStructuredText sources.  ' + default_description)
        writer = Writer()
        reader = Reader()
        cwd = os.getcwd()
        try:
            os.chdir(td)
            output = publish_cmdline_to_binary(reader=reader, writer=writer,
                description=description, argv=['report.rst', 'report.odt'])
            os.chdir(cwd)
            shutil.copy(os.path.join(td, 'report.odt'), filename)
        finally:
            os.chdir(cwd)
