# ============================================================================
#  Search Results Widget 
# ============================================================================

import cv2
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from ui.image_viewer import ImageViewer

class SearchResultsWidget(QWidget):
    def __init__(self, view_post_callback):
        super().__init__()
        self.view_post_callback = view_post_callback
        self.setWindowTitle("Search Results")
        self.setStyleSheet("background-color: #1C1E21; color: white;")

        self.layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("background-color: #1C1E21; color: white;")
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setHorizontalScrollMode(QListWidget.ScrollPerPixel)

        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)

    def display_results(self, results):
        self.list_widget.clear()

        for res in results:
            post = res["post"]
            frame = QFrame()
            frame.setStyleSheet("background-color: #1C1E21; border: 1px solid #4E4F50; border-radius: 5px;")
            layout = QVBoxLayout()

            id_label = QLabel(f"Post ID : {post.get('post_id')} — Similarity: {round(res['similarity'], 4)}")
            id_label.setStyleSheet("font-weight: bold; font-size: 16px; color: white;")
            layout.addWidget(id_label)

            if post.get("images"):
                images_scroll = QScrollArea()
                images_scroll.setWidgetResizable(True)
                images_scroll.setFixedHeight(100)
                images_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                images_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                images_scroll.setStyleSheet("background-color: #1C1E21; border: none;")

                images_widget = QWidget()
                images_layout = QHBoxLayout()
                images_layout.setSpacing(5)
                images_layout.setContentsMargins(0, 0, 0, 0)

                for img_path in post["images"]:
                    if Path(img_path).exists():
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img.shape
                        bytes_per_line = ch * w
                        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pix = QPixmap.fromImage(qimg).scaled(100, 75, Qt.KeepAspectRatio)
                        lbl = QLabel()
                        lbl.setPixmap(pix)
                        lbl.setCursor(Qt.PointingHandCursor)
                        lbl.mousePressEvent = lambda e, p=img_path: self.show_image(p)
                        images_layout.addWidget(lbl)

                images_layout.addStretch()
                images_widget.setLayout(images_layout)
                images_scroll.setWidget(images_widget)
                layout.addWidget(images_scroll)

            view_btn = QPushButton("View Post")
            view_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
            view_btn.clicked.connect(lambda _, pid=post['post_id']: self.view_post_callback(pid))
            layout.addWidget(view_btn)

            frame.setLayout(layout)
            item = QListWidgetItem()
            item.setSizeHint(frame.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, frame)

    def show_image(self, image_path):
        viewer = ImageViewer(image_path)
        viewer.exec_()
