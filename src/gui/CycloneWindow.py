import traceback

from PySide2.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure




class PlotWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding))

        self.left_figure = Figure()
        self.right_figure = Figure()
        self.left_canvas = FigureCanvasQTAgg(self.left_figure)
        self.right_canvas = FigureCanvasQTAgg(self.right_figure)

        self.left_ax = self.left_figure.add_subplot(111)
        self.right_ax = self.right_figure.add_subplot(111)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.left_canvas)
        self.layout.addWidget(self.right_canvas)
        self.setLayout(self.layout)


class CycloneWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = QVBoxLayout()
        self.pickle_selector = PickleSelector()
        self.plot_window = PlotWindow()
        self.setWindowTitle("Cyclone Window")
        self.layout.addWidget(self.pickle_selector)
        self.layout.addWidget(self.plot_window)
        widget = QWidget()
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
