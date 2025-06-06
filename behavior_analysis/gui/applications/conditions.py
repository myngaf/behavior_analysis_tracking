from .application import Application
from PyQt5 import QtWidgets, QtGui, QtCore


class ConditionsWidgets(QtWidgets.QTableWidget):

    def __init__(self, df, conditions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.conditions = [''] + list(conditions)
        # Properties
        self.horizontalHeader().setSectionResizeMode(1)
        # self.setBaseSize(400, 300)
        # self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
        #                                          QtWidgets.QSizePolicy.Expanding))
        # Create table
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["ID", "Date", "Name", "Condition"])
        # Enter animal data
        for idx, animal_info in self.df.iterrows():
            self.append(**animal_info.to_dict())

    def append(self, ID, date, name, condition=None, **kwargs):
        i = self.rowCount()
        self.insertRow(i)
        self.setItem(i, 0, QtWidgets.QTableWidgetItem(ID))
        self.setItem(i, 1, QtWidgets.QTableWidgetItem(date))
        self.setItem(i, 2, QtWidgets.QTableWidgetItem(name))
        combo = QtWidgets.QComboBox()
        combo.addItems(self.conditions)
        self.setCellWidget(i, 3, combo)
        if condition is not None:
            if condition in self.conditions:
                combo.setCurrentIndex(self.conditions.index(condition))

    @property
    def condition_values(self):
        conditions = {}
        for i in range(self.rowCount()):
            ID = self.item(i, 0).text()
            condition = self.cellWidget(i, 3).currentText()
            if len(condition):
                conditions[ID] = condition
            else:
                conditions[ID] = None
        return conditions


class SetConditionsApp(Application):

    def __init__(self, metadata, conditions=(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata  # animal metadata
        self.conditions = conditions  # allowed conditions
        self.name = 'Set animal conditions'
        self.setWindowTitle(self.name)
        self.w = ConditionsWidgets(self.metadata, self.conditions)
        self.add_widget(0, 0, self.w)

    @property
    def output(self):
        return self.w.condition_values
