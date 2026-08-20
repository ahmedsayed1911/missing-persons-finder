# ============================================================================
# Search Widget 
# ============================================================================

import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from config import MAX_IMAGES, SIMILARITY_THRESHOLD
from utils import load_posts, cosine_similarity
from face_model import get_face_embedding_from_image
from ui.image_viewer import ImageViewer


class SearchWidget(QWidget):
    def __init__(self, results_widget, chroma_manager=None):
        super().__init__()
        self.results_widget = results_widget
        self.chroma_manager = chroma_manager
        self.chosen_paths = []
        
        self.setStyleSheet("background-color: #1C1E21; color: white;")
        
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        
        self.select_btn = QPushButton("Select images for search (1-5)")
        self.select_btn.setStyleSheet("background-color: #365899; color: white; font-weight: bold;")
        self.select_btn.clicked.connect(self.select_images)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        self.clear_btn.clicked.connect(self.clear_search)
        
        self.images_preview = QTextEdit()
        self.images_preview.setReadOnly(True)
        self.images_preview.setFixedHeight(60)
        self.images_preview.setStyleSheet("background-color: #242526; color: white;")
        
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.search_btn.clicked.connect(self.perform_search)
        
        top_layout.addWidget(self.select_btn)
        top_layout.addWidget(self.clear_btn)
        top_layout.addWidget(self.images_preview)
        top_layout.addWidget(self.search_btn)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFixedHeight(120)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_scroll.setStyleSheet("background-color: #1C1E21; border: 1px solid #4E4F50;")
        
        self.preview_widget = QWidget()
        self.preview_layout = QHBoxLayout()
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_widget.setLayout(self.preview_layout)
        self.preview_scroll.setWidget(self.preview_widget)
        
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.preview_scroll)
        self.setLayout(main_layout)
    
    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select images", str(Path.cwd()), "Images (*.png *.jpg *.jpeg)"
        )
        if not files:
            return
        
        if len(files) > MAX_IMAGES:
            QMessageBox.warning(self, "Too many images", f"Please select up to {MAX_IMAGES} images.")
            files = files[:MAX_IMAGES]
        
        self.chosen_paths = files
        self.images_preview.setPlainText("\n".join(files))
        
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        for img_path in files:
            if Path(img_path).exists():
                pix = QPixmap(img_path).scaled(100, 75, Qt.KeepAspectRatio)
                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setCursor(Qt.PointingHandCursor)
                lbl.mousePressEvent = lambda e, p=img_path: self.show_image(p)
                self.preview_layout.addWidget(lbl)
        
        self.preview_layout.addStretch()
    
    def show_image(self, image_path):
        viewer = ImageViewer(image_path)
        viewer.exec_()
    
    def perform_search(self):
        if not self.chosen_paths:
            QMessageBox.warning(self, "No images", "Please choose 1 to 5 images for search.")
            return
        
        embeddings = []
        for i, p in enumerate(self.chosen_paths):
            img = cv2.imread(p)
            if img is None:
                print(f"[WARN] Could not read {p}")
                continue
            
            emb = get_face_embedding_from_image(img)
            if emb is not None:
                embeddings.append(emb)
                print(f"[INFO] Extracted embedding {i+1}/{len(self.chosen_paths)}")
        
        if not embeddings:
            QMessageBox.warning(self, "No faces", "No faces detected.")
            return
        
        if self.chroma_manager and self.chroma_manager.get_count() > 0:
            print(f"[INFO] Using ChromaDB for fast search with {len(embeddings)} images...")
            results = self._search_with_chroma(embeddings)
        else:
            print("[INFO] Using linear search (ChromaDB not available)")
            results = self._search_linear(embeddings)
        
        if not results:
            QMessageBox.information(self, "No Matches", "No similar posts found.")
            return
        
        self.results_widget.display_results(results)
    
    def _search_with_chroma(self, embeddings):
        posts = load_posts()
        post_dict = {p['post_id']: p for p in posts}
        results = []
        
        print(f"[DEBUG] ChromaDB has {self.chroma_manager.get_count()} posts")
        print(f"[DEBUG] Total posts in JSON: {len(posts)}")
        
        for i, emb in enumerate(embeddings):
            chroma_results = self.chroma_manager.query_similar(emb, n_results=100)
            
            if chroma_results and chroma_results.get('ids') and len(chroma_results['ids'][0]) > 0:
                print(f"[DEBUG] Found {len(chroma_results['ids'][0])} candidate posts from ChromaDB")
                
                for j, post_id_str in enumerate(chroma_results['ids'][0]):
                    post_id = int(post_id_str.split('_')[1])
                    
                    if post_id not in post_dict:
                        continue
                    
                    post_emb = np.array(post_dict[post_id]['embedding'], dtype=np.float32)
                    similarity = cosine_similarity(emb, post_emb)  
                    
                    print(f"[DEBUG] Post {post_id}: direct cosine similarity = {similarity:.4f}")
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        existing = next((r for r in results if r['post']['post_id'] == post_id), None)
                        
                        if existing:
                            if similarity > existing['similarity']:
                                existing['similarity'] = similarity
                                existing['best_image'] = i + 1
                        else:
                            results.append({
                                'post': post_dict[post_id],
                                'similarity': similarity,
                                'best_image': i + 1
                            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"[INFO] Found {len(results)} matches using ChromaDB + direct cosine similarity")
        return results

    def _search_linear(self, embeddings):
        posts = load_posts()
        if not posts:
            return []
        
        results = []
        print(f"[INFO] Comparing {len(embeddings)} images with {len(posts)} posts...")
        
        for post in posts:
            try:
                post_emb = np.array(post['embedding'], dtype=np.float32)
                
                best_sim = 0.0
                best_img_idx = 0
                
                for i, search_emb in enumerate(embeddings):
                    sim = cosine_similarity(search_emb, post_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_img_idx = i + 1
                
                if best_sim >= SIMILARITY_THRESHOLD:
                    results.append({
                        'post': post,
                        'similarity': best_sim,
                        'best_image': best_img_idx
                    })
            
            except Exception as e:
                print(f"[WARN] Failed to compare with post {post.get('post_id')}: {e}")
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def clear_search(self):
        self.chosen_paths = []
        self.images_preview.clear()
        
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.results_widget.list_widget.clear()