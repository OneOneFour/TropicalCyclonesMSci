import traceback

from PySide2.QtWidgets import *
from PySide2.QtCore import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class BestTrackOptionDialog(QDialog):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options
        self.setWindowTitle("Best Track Options")
        self.layout = QVBoxLayout()

        self.flayout = QFormLayout()
        self.date_to = QDateEdit()
        self.date_to.setDate(options["date_to"])
        self.date_to.setCalendarPopup(True)
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(options["date_from"])
        self.category_from = QSpinBox()
        self.category_from.setValue(options["cat_from"])
        self.category_to = QSpinBox()
        self.category_to.setValue(options["cat_to"])
        self.category_from.setRange(0, 5)
        self.category_to.setRange(0, 5)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.flayout.addRow(QLabel("Date From"), self.date_from)
        self.flayout.addRow(QLabel("Date To"), self.date_to)
        self.flayout.addRow(QLabel("Minimum Category"), self.category_from)
        self.flayout.addRow(QLabel("Maximum Category"), self.category_to)

        tmp_wid = QWidget()
        tmp_wid.setLayout(self.flayout)
        self.layout.addWidget(tmp_wid)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class BestTrackWidget(QWidget):
    def __init__(self, *args,**kwargs):
        super(BestTrackWidget, self).__init__(*args,**kwargs)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        self.layout = QHBoxLayout()
        self.label = QLineEdit()
        self.pushButton = QPushButton("Browse...")
        self.optionButton = QPushButton("Options")
        self.pushButton.clicked.connect(self.get_best_track_file)
        self.optionButton.clicked.connect(self.option_selector)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.pushButton)
        self.layout.addWidget(self.optionButton)
        self.setLayout(self.layout)

    def option_selector(self):
        option_dialog = BestTrackOptionDialog(self.parent().parent().options)
        if option_dialog.exec_():
            options = {
                "date_from":option_dialog.date_from.date(),
                "date_to":option_dialog.date_to.date(),
                "cat_from":option_dialog.category_from.value(),
                "cat_to":option_dialog.category_to.value()
            }
            self.parent().parent().set_options(options)

        else:
            print("Do not save")

    def get_best_track_file(self):
        try:
            file = QFileDialog.getOpenFileName(self, "Open BestTrackCSV", "", "BestTrack Files (*.csv)")
        except Exception as e:
            error = QMessageBox(QMessageBox.Warning,
                                "File save error",
                                "An exception occured during saving",
                                QMessageBox.Ok)
            error.setDetailedText(traceback.format_exc())
            error.exec_()
            return
        if file:
            self.label.setText("".join(file))


class SaveLocationWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        self.layout = QHBoxLayout()
        self.lineEdit = QLineEdit()
        self.pushButton = QPushButton("Browse...")
        self.pushButton.clicked.connect(self.get_save_location)
        self.layout.addWidget(self.lineEdit)
        self.layout.addWidget(self.pushButton)
        self.setLayout(self.layout)

    def get_save_location(self):

        try:
            folder = QFileDialog.getExistingDirectory(self, "Select existing directory")
        except Exception as e:
            error = QMessageBox(QMessageBox.Warning,
                                "File save error",
                                "An exception occured during saving",
                                QMessageBox.Ok)
            error.setDetailedText(traceback.format_exc())
            error.exec_()
            return
        self.lineEdit.setText(folder)


class DownloadBarWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        self.layout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.download_button = QPushButton("Download")
        self.layout.addWidget(self.download_button)
        self.layout.addWidget(self.progressBar)

        self.setLayout(self.layout)


class CycloneWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(CycloneWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Cyclone Window")
        # Set up MPL canvas
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Set up other widgets
        self.__options = {
            "date_from": QDate(2010, 1, 1),
            "date_to": QDate.currentDate(),
            "cat_from": 3,
            "cat_to": 5
        }


        self.layout = QVBoxLayout()
        self.best_track = BestTrackWidget()
        self.save_location = SaveLocationWidget()
        self.download_bar = DownloadBarWidget()
        self.layout.addWidget(self.best_track)
        self.layout.addWidget(self.save_location)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.download_bar)

        main_wid = QWidget()
        main_wid.setLayout(self.layout)
        self.setCentralWidget(main_wid)

    @property
    def options(self):
        return self.__options

    def set_options(self,option):
        self.__options = option