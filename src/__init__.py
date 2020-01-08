from PySide2.QtWidgets import QApplication
from gui.CycloneWindow import CycloneWindow

app = QApplication([])
cyclone_window = CycloneWindow()
cyclone_window.show()
app.exec_()