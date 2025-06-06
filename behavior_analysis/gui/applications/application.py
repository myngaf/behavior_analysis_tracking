from PyQt5 import QtWidgets, QtCore


class Application(QtWidgets.QMainWindow):

    exit_signal = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set central widget
        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(QtWidgets.QVBoxLayout())
        self.statusBar().showMessage('')
        # Create space for user-defined widgets
        self.user_widgets = QtWidgets.QWidget()
        self.user_widgets.setLayout(QtWidgets.QGridLayout())
        self.centralWidget().layout().addWidget(self.user_widgets)
        # Buttons widget
        self.exit_flag = 0
        self.exit_signal.connect(self.close_window)
        self.buttons_widget = QtWidgets.QWidget()
        self.buttons_widget.setLayout(QtWidgets.QHBoxLayout())
        self.buttons_widget.layout().setAlignment(QtCore.Qt.AlignLeft)
        self.buttons = {}
        self._add_button("Confirm", 1)
        self._add_button("Cancel", 0)
        self._add_button("Quit", -1)
        self.centralWidget().layout().addWidget(self.buttons_widget)

    def add_widget(self, i, j, widget=QtWidgets.QWidget, rowspan=1, colspan=1):
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
        self.user_widgets.layout().addWidget(widget, i, j, rowspan, colspan)

    def _add_button(self, name: str, exit_flag: int):
        self.buttons[name] = QtWidgets.QPushButton(name)
        self.buttons[name].setFixedWidth(100)
        self.buttons[name].clicked.connect(lambda x: self.exit_signal.emit(exit_flag))
        self.buttons_widget.layout().addWidget(self.buttons[name])

    @property
    def output(self):
        return

    @QtCore.pyqtSlot(int)
    def close_window(self, i):
        self.exit_flag = i
        self.close()

    @classmethod
    def start(cls, *args, **kwargs) -> tuple:
        """Static method for running the app.

        Returns
        -------
        tuple:
            retval, selected videos, thresholds
        """
        app = QtWidgets.QApplication([])
        window = cls(*args, **kwargs)
        window.show()
        app.exec_()
        exit_flag = window.exit_flag
        output = window.output
        return exit_flag, output
