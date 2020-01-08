from PySide2.QtWidgets import QFrame


class QVLine(QFrame):
    def __init__(self, *args, **kwargs):
        super(QVLine, self).__init__(*args, **kwargs)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class QHLine(QFrame):
    def __init__(self, *args, **kwargs):
        super(QHLine, self).__init__(*args, **kwargs)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
