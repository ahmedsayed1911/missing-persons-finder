# ============================================================================
# Feed Widget 
# ============================================================================

import shutil
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QFrame, QDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from config import KNOWN_DIR, AUTO_REFRESH_MS, SIMILARITY_THRESHOLD
from utils import load_posts, save_posts, cosine_similarity
from ui.image_viewer import ImageViewer

class FeedWidget(QWidget):
    def __init__(self, chroma_manager=None):
        super().__init__()
        self.chroma_manager = chroma_manager
        self.setWindowTitle("Feed")
        self.setStyleSheet("background-color: #1C1E21; color: white;")

        self.layout = QVBoxLayout()

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter Post ID")
        self.search_input.setStyleSheet("background-color: #242526; color: white; border: 1px solid #4E4F50; padding: 3px;")

        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.search_btn.clicked.connect(self.search_by_id)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)
        self.layout.addLayout(search_layout)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("background-color: #1C1E21; color: white;")
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setHorizontalScrollMode(QListWidget.ScrollPerPixel)

        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(AUTO_REFRESH_MS)

        self.refresh()

    def refresh(self):
        
        if not hasattr(self, "list_widget") or self.list_widget is None:
            return
    
        scroll_bar = self.list_widget.verticalScrollBar()
        current_scroll = scroll_bar.value() if scroll_bar else 0
    
        posts = load_posts()
        self.list_widget.clear()
    
        for p in sorted(posts, key=lambda x: x.get("post_id", 0), reverse=True):
            widget = self._create_post_card(p)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            try:
                self.list_widget.addItem(item)
                self.list_widget.setItemWidget(item, widget)
            except RuntimeError:
                return
    
        if scroll_bar:
            QTimer.singleShot(0, lambda: scroll_bar.setValue(current_scroll))


    def _create_post_card(self, post):
        frame = QFrame()
        frame.setStyleSheet("background-color: #1C1E21; border: 1px solid #4E4F50; border-radius: 5px;")
        layout = QVBoxLayout()

        id_label = QLabel(f"Post ID : {post.get('post_id')}")
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
                try:
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
                except Exception as e:
                    print(f"[WARN] error showing image: {e}")

            images_layout.addStretch()
            images_widget.setLayout(images_layout)
            images_scroll.setWidget(images_widget)
            layout.addWidget(images_scroll)

        matches_layout = QVBoxLayout()
        matches = post.get("matches", [])

        if matches:
            for m in matches:
                match_frame = QHBoxLayout()
                match_label = QLabel(f"ID {m['post_id']} — {m['similarity']}")
                match_label.setStyleSheet("color: #ADD8E6; font-weight: bold;")
                view_btn = QPushButton("View Post")
                view_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
                view_btn.clicked.connect(lambda _, pid=m['post_id']: self.show_post_by_id(pid))
                match_frame.addWidget(match_label)
                match_frame.addWidget(view_btn)
                matches_layout.addLayout(match_frame)
        else:
            no_matches_label = QLabel("No matches")
            no_matches_label.setStyleSheet("color: #ADD8E6;")
            matches_layout.addWidget(no_matches_label)

        layout.addLayout(matches_layout)

        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        delete_btn.clicked.connect(lambda _, pid=post['post_id']: self.delete_post(pid))
        layout.addWidget(delete_btn)

        frame.setLayout(layout)
        return frame

    def show_image(self, image_path):
        viewer = ImageViewer(image_path)
        viewer.exec_()

    def search_by_id(self):
        pid_text = self.search_input.text().strip()
        if not pid_text.isdigit():
            QMessageBox.warning(self, "Invalid ID", "Please enter a valid integer Post ID.")
            return

        pid = int(pid_text)
        posts = load_posts()
        post = next((p for p in posts if p['post_id'] == pid), None)

        if not post:
            QMessageBox.information(self, "Not Found", f"Post ID {pid} not found.")
            return

        self.show_post_full(post)

    def show_post_full(self, post):
        dialog = QDialog()
        dialog.setWindowTitle(f"Post {post['post_id']}")
        dialog.setStyleSheet("background-color: #1C1E21;")

        layout = QVBoxLayout()
        frame = QFrame()
        frame.setStyleSheet("background-color: #1C1E21; border: 1px solid #4E4F50; border-radius: 5px;")
        frame_layout = QVBoxLayout()

        id_label = QLabel(f"Post ID : {post.get('post_id')}")
        id_label.setStyleSheet("font-weight: bold; font-size: 16px; color: white;")
        frame_layout.addWidget(id_label)

        if post.get("images"):
            images_scroll = QScrollArea()
            images_scroll.setWidgetResizable(True)
            images_scroll.setFixedHeight(100)
            images_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            images_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            images_widget = QWidget()
            images_layout = QHBoxLayout()

            for img_path in post["images"]:
                if Path(img_path).exists():
                    pix = QPixmap(img_path).scaled(100, 75, Qt.KeepAspectRatio)
                    lbl = QLabel()
                    lbl.setPixmap(pix)
                    lbl.setCursor(Qt.PointingHandCursor)
                    lbl.mousePressEvent = lambda e, p=img_path: self.show_image(p)
                    images_layout.addWidget(lbl)

            images_layout.addStretch()
            images_widget.setLayout(images_layout)
            images_scroll.setWidget(images_widget)
            frame_layout.addWidget(images_scroll)

        matches_layout = QVBoxLayout()
        matches = post.get("matches", [])

        if matches:
            for m in matches:
                match_frame = QHBoxLayout()
                match_label = QLabel(f"ID {m['post_id']} — {m['similarity']}")
                match_label.setStyleSheet("color: #ADD8E6; font-weight: bold;")
                view_btn = QPushButton("View Post")
                view_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
                view_btn.clicked.connect(lambda _, pid=m['post_id']: self.show_post_by_id(pid))
                match_frame.addWidget(match_label)
                match_frame.addWidget(view_btn)
                matches_layout.addLayout(match_frame)
        else:
            no_matches_label = QLabel("No matches")
            no_matches_label.setStyleSheet("color: #ADD8E6;")
            matches_layout.addWidget(no_matches_label)

        frame_layout.addLayout(matches_layout)

        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        delete_btn.clicked.connect(lambda _, pid=post['post_id'], dlg=dialog: self.delete_and_close(pid, dlg))
        frame_layout.addWidget(delete_btn)

        frame.setLayout(frame_layout)
        layout.addWidget(frame)
        dialog.setLayout(layout)
        dialog.resize(800, 600)
        dialog.exec_()

    def show_post_by_id(self, pid):
        posts = load_posts()
        post = next((p for p in posts if p['post_id'] == pid), None)
        if post:
            self.show_post_full(post)

    def delete_and_close(self, post_id, dialog):
        self.delete_post(post_id)
        dialog.accept()

    def delete_post(self, post_id):
        confirm = QMessageBox.question(
            self, "Confirm Delete", f"Delete post ID {post_id}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        posts = load_posts()
        posts = [p for p in posts if p["post_id"] != post_id]

        # Remove from ChromaDB
        if self.chroma_manager is not None:
            try:
                self.chroma_manager.delete_post(post_id)
                print(f"[INFO] Deleted post {post_id} from ChromaDB")
            except Exception as e:
                print(f"[ERROR] Failed to delete from ChromaDB: {e}")

        folder = KNOWN_DIR / str(post_id)
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"[WARN] failed to remove folder: {e}")

        self._recompute_all_matches(posts)

        save_posts(posts)
        self.refresh()
        QMessageBox.information(self, "Deleted", f"Post {post_id} deleted.")

    def _recompute_all_matches(self, posts):
        if self.chroma_manager is not None and self.chroma_manager.get_count() > 0:
            print("[INFO] Using ChromaDB for matching...")
            post_dict = {p['post_id']: p for p in posts}
            
            for p1 in posts:
                try:
                    emb1 = np.array(p1["embedding"], dtype=np.float32)

                    results = self.chroma_manager.query_similar(emb1, n_results=100)

                    matches = []
                    if results and results.get('ids') and len(results['ids'][0]) > 0:
                        for i, post_id_str in enumerate(results['ids'][0]):
                            post_id = int(post_id_str.split('_')[1])

                            if post_id == p1['post_id']:
                                continue

                            emb2 = np.array(post_dict[post_id]['embedding'], dtype=np.float32)
                            similarity = cosine_similarity(emb1, emb2)  # نفس الحساب القديم بالضبط

                            if similarity >= SIMILARITY_THRESHOLD:
                                matches.append({
                                    "post_id": post_id,
                                    "similarity": round(float(similarity), 4)
                                })

                    p1["matches"] = sorted(matches, key=lambda x: x["similarity"], reverse=True)
                    print(f"[INFO] Post {p1['post_id']}: {len(matches)} matches")

                except Exception as e:
                    print(f"[ERROR] Failed to compute matches for post {p1.get('post_id')}: {e}")
                    p1["matches"] = []
        else:
            print("[INFO] Using cosine similarity (no ChromaDB)...")
            for i, p1 in enumerate(posts):
                matches = []
                try:
                    emb1 = np.array(p1["embedding"], dtype=np.float32)
                except Exception:
                    p1["matches"] = []
                    continue

                for j, p2 in enumerate(posts):
                    if i == j:
                        continue

                    try:
                        emb2 = np.array(p2["embedding"], dtype=np.float32)
                    except Exception:
                        continue

                    sim = cosine_similarity(emb1, emb2)
                    if sim >= SIMILARITY_THRESHOLD:
                        matches.append({
                            "post_id": p2["post_id"],
                            "similarity": round(float(sim), 4)
                        })

                p1["matches"] = sorted(matches, key=lambda x: x["similarity"], reverse=True)
                print(f"[INFO] Post {p1['post_id']}: {len(matches)} matches")