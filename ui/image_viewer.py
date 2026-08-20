# ============================================================================
# Image Viewer Dialog
# ============================================================================

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


class ImageViewer(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setModal(True)
        self.setStyleSheet("background-color: #1C1E21;")
        
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        lbl = QLabel()
        pix = QPixmap(image_path)
        lbl.setPixmap(pix)
        lbl.setAlignment(Qt.AlignCenter)
        
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.resize(800, 600)
