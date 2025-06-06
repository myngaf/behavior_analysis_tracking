from PyQt5 import QtWidgets


class FileListWidget(QtWidgets.QListWidget):

    def __init__(self, files: list, min_width=150, max_width=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = files
        # Add files to list
        self.addItems([f.name for f in self.files])
        self.setCurrentRow(0)
        # Set size
        self.setMinimumWidth(min_width)
        self.setMaximumWidth(max_width)
