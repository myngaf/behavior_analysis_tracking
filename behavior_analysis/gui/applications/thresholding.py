from video_analysis_toolbox.utilities import TrackingError
from video_analysis_toolbox.video import Video, FrameErrorWarning
from behavior_analysis.tracking import Tracker

from PyQt5 import QtWidgets, QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import warnings

from ..widgets.slider import SliderWidget


class SetThresholdsApp(QtWidgets.QMainWindow):
    """Application for setting thresholds for tracking.

    Parameters
    ----------
    tracker : Tracker
        A Tracker object.
    videos : list
        List of paths to video files.
    """

    def __init__(self, tracker, videos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.videos = [Video.open(video) for video in videos]
        # Set name
        self.name = 'Thresholding'
        self.setWindowTitle(self.name)
        # Resize window
        self.resize(800, 800)
        # Set main widget
        self.widget = ThresholdWidget(parent=self)
        self.layout = QtWidgets.QGridLayout()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.statusBar().showMessage('')

    @staticmethod
    def start(tracker: Tracker, videos: list) -> tuple:
        """Static method for running the app.

        Returns
        -------
        tuple:
            threshold for finding fish, threshold for detecting internal features
        """
        app = QtWidgets.QApplication([])
        window = SetThresholdsApp(tracker, videos)
        window.show()
        app.exec_()
        thresholds = (window.tracker.fish_detector.threshold, window.tracker.feature_detector.threshold)
        return thresholds


class ThresholdWidget(QtWidgets.QWidget):
    """Widget for setting thresholds.

    Parameters
    ----------
    parent : QtWidgets.QMainWindow
        Parent widget must have a tracker attribute.
    """

    def __init__(self, parent: QtWidgets.QMainWindow, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)

        # Set layout
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        # Set current video and initialize frame number and images
        self.video = self.parent().videos[0]
        self.frame_number = 0
        self.images = (None, None, None, None)

        # Create display widget for showing images
        self.display_widget = self.add_widget(1, 0, layout=QtWidgets.QVBoxLayout)

        # Create combobox for switching between images
        self.box_widget = QtWidgets.QComboBox()
        self.box_widget.addItems(['Input image', 'Thresholded', 'Contours', 'Tracking'])
        self.box_widget.currentIndexChanged.connect(self.change_display_image)
        self.box_widget.setFixedSize(120, 25)
        self.display_widget.layout().addWidget(self.box_widget, alignment=QtCore.Qt.AlignRight)

        # Create figure widget
        self.figure = plt.figure(facecolor='0.95')
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(500, 500)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy().MinimumExpanding, QtWidgets.QSizePolicy().MinimumExpanding)
        self.display_widget.layout().addWidget(self.canvas)

        # Initialize the display
        self.update_display_image()
        self.display_index = 0
        self.image_ = self.ax.imshow(self.images[self.display_index],
                                     origin='upper',
                                     cmap='Greys_r',
                                     vmin=0, vmax=255)

        # Create slider widget for changing frame and thresholds
        self.slider_widget = self.add_widget(0, 0, layout=QtWidgets.QVBoxLayout)
        self.slider_widget.layout().setSpacing(0)
        # Frame slider
        self.frame_slider = SliderWidget('Frame', 0, self.video.frame_count - 1, 0)
        self.frame_slider.value_changed.connect(self.change_frame)
        self.slider_widget.layout().addWidget(self.frame_slider)
        # Threshold sliders
        thresh1 = self.parent().tracker.fish_detector.threshold
        thresh2 = self.parent().tracker.feature_detector.threshold
        self.thresh1_widget = SliderWidget('Thresh 1', 0, 254, thresh1)
        self.thresh2_widget = SliderWidget('Thresh 2', 1, 255, thresh2)
        self.thresh1_widget.value_changed.connect(self.change_thresh1)
        self.thresh2_widget.value_changed.connect(self.change_thresh2)
        self.slider_widget.layout().addWidget(self.thresh1_widget)
        self.slider_widget.layout().addWidget(self.thresh2_widget)

        # Create video widget for switching between videos
        self.video_widget = self.add_widget(0, 1, rowspan=2)
        self.video_widget.setMinimumWidth(150)
        self.video_widget.setMaximumWidth(200)
        self.video_list = QtWidgets.QListWidget()
        self.video_widget.layout().addWidget(self.video_list)
        # Add videos to list
        self.video_list.addItems([video.name for video in self.parent().videos])
        self.video_list.itemSelectionChanged.connect(self.switch_video)
        self.video_list.setCurrentRow(0)

    def add_widget(self, i, j, widget=QtWidgets.QWidget, layout: QtWidgets.QLayout = QtWidgets.QGridLayout,
                   rowspan=1, colspan=1):
        """Adds a new widget to the grid.

        Parameters
        ----------
        i, j : int
            Row number, column number.
        widget : QtWidgets.QWidget type
            The type of widget to add.
        layout : QtWidgets.QLayout type
            The layout type to use.
        rowspan, colspan : int
            Number of rows in grid widget should span, number of columns in grid widget should span.

        Returns
        -------
        QtWidgets.QWidget
            The newly created widget.
        """
        w = widget()
        w.setLayout(layout())
        self.layout.addWidget(w, i, j, rowspan, colspan)
        return w

    def draw(self):
        """Redraws the display image in the GUI."""
        self.image_.set_data(self.images[self.display_index])
        self.canvas.draw()
        self.canvas.flush_events()

    def update_display_image(self):
        """Updates the display image. Called whenever GUI state changes (e.g. frame changes, thresholds change, new
        image selected)."""
        with warnings.catch_warnings(record=True) as w:  # catch frame warnings so that GUI does not crash
            warnings.simplefilter("always")
            image = self.video.grab_frame(self.frame_number)  # grab the current frame
            w = list(filter(lambda i: issubclass(i.category, FrameErrorWarning), w))
            if len(w):
                self.parent().statusBar().showMessage(str(w[0].message), 1000)  # show any warning in the status bar
            else:
                try:  # catch tracking errors
                    masked, fish, features = self.parent().tracker.find_contours(image)  # find contours
                    feature_info = self.parent().tracker.track_points(features)  # get tracking info
                    centre = (feature_info['sb'].x, feature_info['sb'].y)  # get swim bladder centre
                    masked[masked > 0] = 1  # binarize fish mask
                    masked = self.parent().tracker.draw_mask(image, masked)
                    tail_points = self.parent().tracker.tail_tracker.track_tail(masked, centre)  # track tail
                    feature_info['tail_points'] = tail_points  # add tail points to feature info
                    contoured = self.parent().tracker.draw_contours(image, fish, features)  # show contours
                    tracked = self.parent().tracker.draw_tracking(image, **feature_info)  # show tracking
                except TrackingError:  # if there was an error, don't show tracking and display a message in status bar
                    masked = np.zeros(image.shape, dtype='uint8')
                    contoured = image
                    tracked = image
                    self.parent().statusBar().showMessage('Cannot detect contours - adjust thresholds', 1000)
                self.images = (image, masked, contoured, tracked)

    @QtCore.pyqtSlot()
    def switch_video(self):
        """Switches between videos."""
        selected_video_index = self.video_list.currentRow()  # get the currently selected row of the video list
        self.video = self.parent().videos[selected_video_index]  # set the new video
        self.frame_slider.set_range(0, self.video.frame_count)  # reset frame slider range to fit new video
        self.frame_slider.set_value(0)  # go to first frame of video

    @QtCore.pyqtSlot(int)
    def change_display_image(self, i):
        """Changes the image to be displayed (e.g. contours, tracking etc.)."""
        self.display_index = i
        self.draw()

    @QtCore.pyqtSlot(int)
    def change_frame(self, frame):
        """Called when the frame changes."""
        self.frame_number = frame
        self.update_display_image()
        self.draw()

    @QtCore.pyqtSlot(int)
    def change_thresh1(self, val):
        """Called when thresh1 changes."""
        thresh2 = self.thresh2_widget.value
        if thresh2 <= val:
            self.thresh2_widget.set_value(val + 1)
        self._change_thresholds()

    @QtCore.pyqtSlot(int)
    def change_thresh2(self, val):
        """Called when thresh2 changes."""
        thresh1 = self.thresh1_widget.value
        if thresh1 >= val:
            self.thresh1_widget.set_value(val - 1)
        self._change_thresholds()

    def _change_thresholds(self):
        """Called whenever either threshold changes. Updates thresholds in parent tracker then updates the display."""
        self.parent().tracker.fish_detector.threshold = self.thresh1_widget.value
        self.parent().tracker.feature_detector.threshold = self.thresh2_widget.value
        self.update_display_image()
        self.draw()
