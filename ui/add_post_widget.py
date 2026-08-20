# ============================================================================
# Add Post Widget 
# ============================================================================


import uuid
import cv2
import numpy as np
import threading
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtCore import QTimer
from config import KNOWN_DIR, MAX_IMAGES, SIMILARITY_THRESHOLD
from utils import load_posts, save_posts, cosine_similarity
from face_model import images_to_embedding_list, get_face_app

class AddPostWidget(QWidget):
    def __init__(self, on_post_added, chroma_manager=None):
        super().__init__()
        self.on_post_added = on_post_added
        self.chroma_manager = chroma_manager
        self.chosen_paths = []

        self.setWindowTitle("Add Post")
        self.setStyleSheet("background-color: #1C1E21; color: white;")

        self.id_label = QLabel("Post ID")
        self.id_label.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Enter unique integer ID")
        self.id_input.setStyleSheet("background-color: #242526; border: 1px solid #4E4F50; padding: 3px; color: white;")

        self.select_btn = QPushButton("Select images (1-5)")
        self.select_btn.setStyleSheet("background-color: #365899; color: white; font-weight: bold;")
        self.select_btn.clicked.connect(self.select_images)

        self.images_preview = QTextEdit()
        self.images_preview.setReadOnly(True)
        self.images_preview.setFixedHeight(80)
        self.images_preview.setStyleSheet("background-color: #242526; color: white;")

        self.add_btn = QPushButton("Add Post")
        self.add_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.add_btn.clicked.connect(self.add_post)

        layout = QVBoxLayout()
        layout.addWidget(self.id_label)
        layout.addWidget(self.id_input)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.images_preview)
        layout.addWidget(self.add_btn)
        self.setLayout(layout)

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select images", str(Path.cwd()), "Images (*.png *.jpg *.jpeg)")
        if not files:
            return

        if len(files) > MAX_IMAGES:
            QMessageBox.warning(self, "Too many images", f"Please select up to {MAX_IMAGES} images.")
            files = files[:MAX_IMAGES]

        self.chosen_paths = files
        self.images_preview.setPlainText("\n".join(files))

    def add_post(self):
        if get_face_app() is None:
            QMessageBox.warning(self, "Face Model Not Ready", "Face model is still loading. Please wait...")
            return

        post_id_text = self.id_input.text().strip()
        if not post_id_text.isdigit():
            QMessageBox.warning(self, "Invalid ID", "Post ID must be an integer.")
            return

        posts = load_posts()
        if any(p["post_id"] == int(post_id_text) for p in posts):
            QMessageBox.warning(self, "Duplicate ID", f"Post ID {post_id_text} already exists.")
            return

        if len(self.chosen_paths) == 0:
            QMessageBox.warning(self, "No images", "Please choose 1 to 5 images.")
            return

        threading.Thread(
            target=self._process_and_save_post,
            args=(int(post_id_text), list(self.chosen_paths)),
            daemon=True
        ).start()

        QMessageBox.information(self, "Processing", "Post is being processed.")
        self.id_input.clear()
        self.images_preview.clear()
        self.chosen_paths = []

    def _process_and_save_post(self, post_id, image_paths):
        try:
            print(f"[INFO] Processing post {post_id}...")
            emb = images_to_embedding_list(image_paths)

            if emb is None:
                def show_err():
                    QMessageBox.critical(self, "No faces", "No faces detected.")
                QTimer.singleShot(0, show_err)
                return

            print(f"[INFO] Embedding extracted for post {post_id}")

            post_folder = KNOWN_DIR / str(post_id)
            post_folder.mkdir(parents=True, exist_ok=True)

            saved_paths = []
            for src in image_paths:
                ext = Path(src).suffix
                dst = post_folder / (str(uuid.uuid4()) + ext)
                img = cv2.imread(src)
                if img is not None:
                    try:
                        cv2.imwrite(str(dst), img)
                        saved_paths.append(str(dst))
                    except Exception as e:
                        print(f"[WARN] failed to save image {src}: {e}")

            print(f"[INFO] Saved {len(saved_paths)} images for post {post_id}")

            posts = load_posts()
            new_post = {
                "post_id": post_id,
                "images": saved_paths,
                "embedding": emb.tolist(),
                "matches": []
            }
            posts.append(new_post)

            if self.chroma_manager is not None:
                try:
                    self.chroma_manager.add_post(
                        post_id=post_id,
                        embedding=emb,
                        metadata={'num_images': len(saved_paths)}
                    )
                    print(f"[INFO] Added post {post_id} to ChromaDB")
                except Exception as e:
                    print(f"[ERROR] Failed to add to ChromaDB: {e}")

            print(f"[INFO] Computing matches for post {post_id}...")
            self._recompute_all_matches(posts)

            save_posts(posts)

            print(f"[INFO] Post {post_id} added successfully!")

            if self.on_post_added:
                QTimer.singleShot(0, self.on_post_added)

        except Exception as e:
            print(f"[ERROR] Failed to process post: {e}")
            import traceback
            traceback.print_exc()

            def show_err():
                QMessageBox.critical(self, "Error", f"Failed to process post: {e}")
            QTimer.singleShot(0, show_err)

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
                            similarity = cosine_similarity(emb1, emb2) 
                            
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