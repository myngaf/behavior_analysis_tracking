from .application import Application
from ..widgets import MultiSliderWidget, FileListWidget
from behavior_analysis.tracking.segmentation import BoutDetector
from PyQt5 import QtCore
import pyqtgraph as pg
import pandas as pd


class BoutPlottingWidget(pg.GraphicsLayoutWidget):

    pens = dict(black=pg.mkPen('k', width=1), red=pg.mkPen('r', width=2))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p0 = self.addPlot(row=0, col=0)
        self.p1 = self.addPlot(row=1, col=0)
        self.p1.setXLink(self.p0)
        for p in (self.p0, self.p1):
            p.showButtons()
            p.setAutoPan(x=True)
            p.setAspectLocked(False)

    def update(self, y0, y1, thresh, bouts, frame_rate):
        x = y0.index / frame_rate
        ylim = y0.abs().max()
        ylim = ylim + (0.1 * ylim)
        # Update tail angle plot
        self.p0.plot(x, y0, pen=self.pens['black'], clear=True)
        self.p0.setLabel(axis='left', text='Tail angle [rad]')
        # Update filtered derivative plot
        self.p1.plot(x, y1, pen=self.pens['black'], clear=True)
        self.p1.plot([x[0], x[-1]], [thresh, thresh], pen=self.pens['red'])
        self.p1.setLabel(axis='left', text='Convolved tail angle [rad]')
        self.p1.setLabel(axis='bottom', text='Time [s]')
        # Label bouts
        for c, (start, end) in enumerate(bouts):
            lower = pg.PlotDataItem(x[(start, end),], [-ylim, -ylim])
            upper = pg.PlotDataItem(x[(start, end),], [ylim, ylim])
            fill = pg.FillBetweenItem(lower, upper, brush=pg.intColor(c, alpha=150))
            self.p0.addItem(fill)
        self.p0.setLimits(xMin=x[0], xMax=x[-1], yMin=-ylim, yMax=ylim, minYRange=2 * ylim)
        self.p0.setRange(yRange=(-ylim, ylim))
        self.p1.setLimits(xMin=x[0], xMax=x[-1], yMin=0, yMax=y1.max(), minYRange=y1.max())
        self.p1.setRange(yRange=(0, y1.max()))


class BoutDetectionThresholdApp(Application):

    def __init__(self, metadata, threshold=0.02, winsize=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Open files
        self.metadata = metadata
        self._filenames = self.metadata['path'].values
        self.data = {}
        # Set name
        self.setWindowTitle('Set bout detection thresholds')
        # Resize window
        self.resize(1600, 800)

        # Sliders
        self.sliders_widget = MultiSliderWidget()
        self.sliders_widget.add_slider('Threshold', 0, 1.0, threshold, decimals=3)
        self.sliders_widget.add_slider('Kernel size', 0, 2.0, winsize)
        self.add_widget(0, 0, self.sliders_widget)
        for (name, slider) in self.sliders_widget.sliders.items():
            slider.value_changed.connect(self.change_thresholds)
        # Plotting
        pg.setConfigOptions(foreground=0.0, background=1.0)
        self.plots = BoutPlottingWidget()
        self.add_widget(1, 0, self.plots)
        # File list
        self.files_widget = FileListWidget(self._filenames)
        self.add_widget(0, 1, self.files_widget, rowspan=2)
        self.files_widget.itemSelectionChanged.connect(self.update_plots)

        # Initialize plots
        data, metadata = self.current
        self.bout_detector = BoutDetector(threshold, winsize, metadata.fps)
        self.update_plots()

    @property
    def current(self):
        idx = self.files_widget.currentRow()
        metadata = self.metadata.iloc[idx]
        name = self._filenames[idx]
        try:
            data = self.data[name]
        except KeyError:
            self.statusBar().showMessage(f'Opening file {name}...')
            data = pd.read_csv(metadata.path).loc[:, 'tip']
            self.data[name] = data
            self.statusBar().clearMessage()
        return data, metadata

    @QtCore.pyqtSlot(float)
    def change_thresholds(self, val):
        threshold = self.sliders_widget.sliders['Threshold'].value
        winsize = self.sliders_widget.sliders['Kernel size'].value
        self.bout_detector.threshold = threshold
        self.bout_detector.winsize = max(0.001, winsize)
        self.update_plots()

    @QtCore.pyqtSlot()
    def update_plots(self):
        data, metadata = self.current
        bouts = self.bout_detector(data, frame_rate=metadata.fps)
        threshold = self.bout_detector.threshold
        filtered = self.bout_detector._filtered
        self.plots.update(data, filtered, threshold, bouts, metadata.fps)

    @property
    def output(self):
        return self.bout_detector.threshold, self.bout_detector.winsize
