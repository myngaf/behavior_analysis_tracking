from behavior_analysis.gui.widgets.file_list import FileListWidget, FigureWidget
from video_analysis_toolbox.gui.apps.userapp import UserApplicationWindow
from PyQt5 import QtWidgets, QtCore
import pandas as pd


class KinematicsWidget(QtWidgets.QWidget):

    def __init__(self, files, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLayout(QtWidgets.QGridLayout())
        # Figure
        self.plot = FigureWidget()
        self.layout().addWidget(self.plot)
        # File list
        self.files = files
        self.file_list = FileListWidget(self.files)
        self.data_frames = {}
        self.layout().addWidget(self.file_list, 0, 1)
        self.file_list.list.currentRowChanged.connect(self.update_plots)
        self.update_plots(0)

    @QtCore.pyqtSlot(int)
    def update_plots(self, i):
        current = self.files[i]
        try:
            df = self.data_frames[current.name]
        except KeyError:
            df = pd.read_csv(current)
            df = df[df['tracked']]
            self.data_frames[current.name] = df
        self.plot.ax.clear()
        self.plot.ax.plot(df.loc[:, 'tip'])
        self.plot.ax.plot(df.loc[:, 'right_angle'] - df.loc[:, 'left_angle'])
        self.plot.canvas.draw()
        self.plot.canvas.flush_events()


class KinematicsBrowser(UserApplicationWindow):

    def __init__(self, files, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = files
        # Set name
        self.setWindowTitle('Kinematics Browser')
        # Resize window
        self.resize(800, 800)
        # Threshold widget
        self.kinematics_widget = KinematicsWidget(self.files)
        self.centralWidget().layout().insertWidget(0, self.kinematics_widget)

    @property
    def output(self):
        return self.kinematics_widget.file_list.current
