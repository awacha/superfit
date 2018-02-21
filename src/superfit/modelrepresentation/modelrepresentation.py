from PyQt5 import QtWidgets
from .modelrepresentation_ui import Ui_Form
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg

class ModelRepresentation(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

    def setupUi(self, Form):
        super().setupUi(Form)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.figureToolBar = NavigationToolbar2QT(self.canvas, self)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(self.figureToolBar)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)

