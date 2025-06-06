from PyQt5 import QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PlottingWidget(QtWidgets.QWidget):

    xmin = 500
    ymin = 500

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setLayout(QtWidgets.QGridLayout())
        # Create figure widget
        self.figure, self.axes = plt.subplots(*args, **kwargs)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(self.xmin, self.ymin)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy().MinimumExpanding, QtWidgets.QSizePolicy().MinimumExpanding)
        self.layout().addWidget(self.canvas)

    def draw(self):
        self.canvas.draw()
        self.canvas.flush_events()

    def clear(self):
        for ax in self.axes:
            ax.clear()
