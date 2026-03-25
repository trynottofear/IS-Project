import sys
import cv2
import numpy as np
import time
import math
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QFileDialog, QListWidget, QDialog, QMessageBox,
    QTabWidget, QScrollArea, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon

from db_manager import DatabaseManager
from face_pipeline import FaceProcessor

# --- Stylesheet ---
DARK_THEME_QSS = """
    QMainWindow, QDialog { background-color: #1e1e2e; color: #cdd6f4; }
    QLabel { color: #cdd6f4; font-family: 'Segoe UI', Arial, sans-serif; }
    
    /* Tabs */
    QTabWidget::pane { border: 1px solid #313244; background-color: #1e1e2e; }
    QTabBar::tab { background-color: #181825; color: #a6adc8; padding: 12px 25px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; }
    QTabBar::tab:selected { background-color: #313244; color: #cdd6f4; }
    QTabBar::tab:hover { background-color: #45475a; }
    
    /* Buttons */
    QPushButton { 
        background-color: #89b4fa; color: #11111b; 
        border-radius: 5px; padding: 8px 15px; font-weight: bold; font-family: 'Segoe UI', Arial, sans-serif;
    }
    QPushButton:hover { background-color: #b4befe; }
    QPushButton:pressed { background-color: #74c7ec; }
    
    QPushButton#danger { background-color: #f38ba8; }
    QPushButton#danger:hover { background-color: #eba0ac; }
    
    QPushButton#success { background-color: #a6e3a1; }
    QPushButton#success:hover { background-color: #94e2d5; }
    
    QPushButton#secondary { background-color: #cba6f7; }
    QPushButton#secondary:hover { background-color: #f5c2e7; }
    
    /* Inputs */
    QLineEdit, QComboBox { 
        background-color: #181825; color: #cdd6f4; border: 1px solid #313244; 
        padding: 8px; border-radius: 4px; font-size: 14px;
    }
    QLineEdit:focus, QComboBox:focus { border: 1px solid #89b4fa; }
    QComboBox::drop-down { border: none; }
    
    /* Scroll Areas & Lists */
    QScrollArea { border: none; background-color: transparent; }
    QScrollBar:vertical { background: #181825; width: 10px; margin: 0px; }
    QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 20px; }
"""

# --- Threads
class FaceVideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list)
    log_updated = pyqtSignal() # Emitted when VIP/Blacklist is logged
    
    def __init__(self, face_processor, db):
        super().__init__()
        self.face_processor = face_processor
        self.db = db
        self.running = True
        self.live_tracks = []
        self.next_track_id = 1
        self.last_logged_identities = {} # Maps {identity_id: timestamp_float}

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # To prevent overloading, we can process on every frame if fast enough, 
                # or every N frames. With GPU, every frame is usually ok.
                results = self.face_processor.process_frame(frame)
                
                current_boxes = [r['box'] for r in results]
                unmatched_boxes = set(range(len(current_boxes)))
                
                target_tracks = []
                for track in self.live_tracks:
                    track['misses'] += 1
                    best_score = 0
                    best_idx = -1
                    for i in unmatched_boxes:
                        score = compute_tracking_score(track['last_box'], current_boxes[i])
                        if score > best_score:
                            best_score = score
                            best_idx = i
                            
                    if best_score > 0.4:
                        res = results[best_idx]
                        track['last_box'] = current_boxes[best_idx]
                        track['misses'] = 0
                        
                        # Robust continuous Confidence Latching
                        if 'history' not in track:
                            track['history'] = []
                            
                        # Gather votes at a lower threshold (0.65) since tracking provides temporal stability
                        if res['category'] != 'Unknown' and res['similarity'] > 0.65:
                            pred = (res['identity_id'], res['name'], res['category'])
                        else:
                            pred = (None, 'Unknown', 'Unknown')
                            
                        track['history'].append(pred)
                        if len(track['history']) > 30: # Rolling window of 30 frames
                            track['history'].pop(0)
                        
                        # Find known person votes in history
                        known_preds = [p for p in track['history'] if p[1] != 'Unknown']
                        if known_preds:
                            votes = {}
                            for p in known_preds:
                                votes[p] = votes.get(p, 0) + 1
                            # Max tuple by vote count
                            most_common_pred = max(votes.items(), key=lambda x: x[1])
                            best_pred, count = most_common_pred
                            
                            # If we are currently Unknown, require 3 matches to confidently latch
                            if track['name'] == 'Unknown':
                                if count >= 3:
                                    track['identity_id'] = best_pred[0]
                                    track['name'] = best_pred[1]
                                    track['category'] = best_pred[2]
                            else:
                                # We are latched. Only swap tracks if a DIFFERENT person gets 5 matches recently
                                if best_pred[1] != track['name'] and count >= 5:
                                    track['identity_id'] = best_pred[0]
                                    track['name'] = best_pred[1]
                                    track['category'] = best_pred[2]
                            
                        unmatched_boxes.remove(best_idx)
                        
                    if track['misses'] < 5:
                        target_tracks.append(track)
                        
                # New trajectories
                for i in unmatched_boxes:
                    res = results[i]
                    if res['category'] != 'Unknown' and res['similarity'] > 0.65:
                        pred = (res['identity_id'], res['name'], res['category'])
                    else:
                        pred = (None, 'Unknown', 'Unknown')
                        
                    target_tracks.append({
                        'id': self.next_track_id,
                        'last_box': current_boxes[i],
                        'misses': 0,
                        'identity_id': None,
                        'name': 'Unknown',
                        'category': 'Unknown',
                        'last_logged_time': 0.0,
                        'history': [pred]
                    })
                    self.next_track_id += 1
                    
                self.live_tracks = target_tracks
                
                now = time.time()
                log_triggered = False
                display_data = []
                for t in self.live_tracks:
                    if t['misses'] == 0:
                        display_data.append({
                            'box': t['last_box'],
                            'name': t['name'],
                            'category': t['category']
                        })
                        
                        # Logic to log VIP and Blacklists to DB globally (10 mins lag)
                        if t['category'] in ['VIP', 'Blacklist'] and t['identity_id'] is not None:
                            last_t = self.last_logged_identities.get(t['identity_id'], 0.0)
                            if now - last_t > 600:
                                self.db.log_detection(t['identity_id'], t['name'], t['category'])
                                self.last_logged_identities[t['identity_id']] = now
                                log_triggered = True
                                
                if log_triggered:
                    self.log_updated.emit()
                    
                self.frame_ready.emit(frame, display_data)
            else:
                break

    def stop(self):
        self.running = False
        self.wait()
        if self.cap:
            self.cap.release()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, face_processor):
        super().__init__()
        self.face_processor = face_processor
        self._run_flag = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                results = self.face_processor.process_frame(frame)
                for res in results:
                    box = res['box']
                    name = res['name']
                    category = res['category']
                    similarity = res['similarity']
                    
                    if category == 'VIP':
                        color = (0, 255, 0)
                    elif category == 'Blacklist':
                        color = (0, 0, 255)
                    else:
                        color = (128, 128, 128)
                        
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label_text = f"{name} ({category}) - {similarity:.2f}" if name != "Unknown" else "Unknown"
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self._run_flag = False
        self.wait()
        if self.cap:
            self.cap.release()

class CaptureDialog(QDialog):
    """ Dialog to capture photo from webcam """
    captured_image = pyqtSignal(str) # Emits the save path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Photo")
        self.setFixedSize(600, 500)
        
        layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(580, 400)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        self.btn_capture = QPushButton("Take Snapshot")
        self.btn_capture.setObjectName("success")
        self.btn_capture.clicked.connect(self.take_snapshot)
        layout.addWidget(self.btn_capture)
        
        self.setLayout(layout)
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.current_frame = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qt_img).scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(pix)

    def take_snapshot(self):
        if self.current_frame is not None:
            # Save frame temporarily
            import tempfile
            import os
            temp_path = os.path.join(tempfile.gettempdir(), f"capture_{np.random.randint(1000)}.jpg")
            cv2.imwrite(temp_path, self.current_frame)
            self.captured_image.emit(temp_path)
            self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

class AddEditIdentityDialog(QDialog):
    def __init__(self, db_manager, face_processor, identity_data=None, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.face_processor = face_processor
        self.identity_data = identity_data # None for new, dict for edit
        self.setWindowTitle("Manage Identity" if self.identity_data else "Add New Identity")
        self.setMinimumSize(500, 600)
        
        # List to hold temporary paths to images before saving
        self.pending_images = []
        
        # UI Setup
        layout = QVBoxLayout()
        
        form_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full Name")
        if self.identity_data:
            self.name_input.setText(self.identity_data['name'])
            
        self.category_combo = QComboBox()
        # Only show Unknown if it's already an Unknown, or if we need to let them upgrade unknown to VIP
        if self.identity_data and self.identity_data.get('category') == 'Unknown':
            self.category_combo.addItems(["Unknown", "VIP", "Blacklist"])
            self.category_combo.setCurrentText("Unknown")
        else:
            self.category_combo.addItems(["VIP", "Blacklist"])
            if self.identity_data:
                self.category_combo.setCurrentText(self.identity_data['category'])
            
        form_layout.addWidget(self.name_input, stretch=2)
        form_layout.addWidget(self.category_combo, stretch=1)
        layout.addLayout(form_layout)
        
        # Images Gallery Setup
        gallery_label = QLabel("Reference Photos")
        font = QFont()
        font.setBold(True)
        gallery_label.setFont(font)
        layout.addWidget(gallery_label)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.scroll_area.setWidget(self.gallery_widget)
        layout.addWidget(self.scroll_area)
        
        btn_layout = QHBoxLayout()
        
        self.btn_upload = QPushButton("Upload Files")
        self.btn_upload.clicked.connect(self.upload_images)
        btn_layout.addWidget(self.btn_upload)
        
        self.btn_camera = QPushButton("Take Photo via Camera")
        self.btn_camera.setObjectName("secondary")
        self.btn_camera.clicked.connect(self.open_camera_capture)
        btn_layout.addWidget(self.btn_camera)
        
        layout.addLayout(btn_layout)
        
        # Save Button
        self.btn_save = QPushButton("Save & Enroll")
        self.btn_save.setObjectName("success")
        self.btn_save.clicked.connect(self.save_identity)
        layout.addWidget(self.btn_save)
        
        self.setLayout(layout)
        self.refresh_gallery()

    def upload_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_names:
            self.pending_images.extend(file_names)
            self.refresh_gallery()

    def open_camera_capture(self):
        cap_dialog = CaptureDialog(self)
        cap_dialog.captured_image.connect(self.add_captured_image)
        cap_dialog.exec()
        
    def add_captured_image(self, path):
        self.pending_images.append(path)
        self.refresh_gallery()

    def remove_pending_image(self, index):
        if 0 <= index < len(self.pending_images):
            self.pending_images.pop(index)
            self.refresh_gallery()

    def refresh_gallery(self):
        # Clear existing
        for i in reversed(range(self.gallery_layout.count())): 
            widget = self.gallery_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                
        col = 0
        row = 0
        
        # Show existing images if editing
        if self.identity_data and 'embeddings' in self.identity_data:
            for emb in self.identity_data['embeddings']:
                frame = self.create_thumbnail_frame(emb['image_path'], is_existing=True, emb_id=emb['id'])
                if frame:
                    self.gallery_layout.addWidget(frame, row, col)
                    col += 1
                    if col > 2:
                        col = 0
                        row += 1

        # Show pending images
        for idx, img_path in enumerate(self.pending_images):
            frame = self.create_thumbnail_frame(img_path, is_existing=False, index=idx)
            if frame:
                self.gallery_layout.addWidget(frame, row, col)
                col += 1
                if col > 2:
                    col = 0
                    row += 1

    def create_thumbnail_frame(self, path, is_existing=False, index=None, emb_id=None):
        frame = QFrame()
        frame.setStyleSheet("background-color: #313244; border-radius: 5px;")
        frame.setFixedSize(140, 160)
        vbox = QVBoxLayout(frame)
        
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            pixmap = QPixmap(path).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio)
            img_label.setPixmap(pixmap)
        except:
            img_label.setText("Invalid Image")
            
        vbox.addWidget(img_label)
        
        # Delete button
        del_btn = QPushButton("Remove")
        del_btn.setObjectName("danger")
        if is_existing:
            del_btn.clicked.connect(lambda _, eid=emb_id: self.delete_existing_embedding(eid))
        else:
            del_btn.clicked.connect(lambda _, i=index: self.remove_pending_image(i))
            
        vbox.addWidget(del_btn)
        return frame

    def delete_existing_embedding(self, emb_id):
        confirm = QMessageBox.question(self, "Confirm Delete", "Are you sure you want to delete this reference photo?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.db.delete_embedding(emb_id)
            # Refetch data to update ui
            all_ids = self.db.get_all_identities_with_embeddings()
            for ident in all_ids:
                if ident['id'] == self.identity_data['id']:
                    self.identity_data = ident
                    break
            self.refresh_gallery()

    def save_identity(self):
        name = self.name_input.text().strip()
        category = self.category_combo.currentText()
        
        if not name:
            QMessageBox.warning(self, "Error", "Name cannot be empty.")
            return
            
        # For new users, we MUST have at least one image. 
        # For existing users, they MUST have at least one image in total (either pending or existing).
        total_images = len(self.pending_images)
        if self.identity_data and 'embeddings' in self.identity_data:
            total_images += len(self.identity_data['embeddings'])
            
        if total_images == 0:
            QMessageBox.warning(self, "Error", "An identity must have at least one reference photo.")
            return

        # Process new images and get embeddings
        processed_data = []
        for path in self.pending_images:
            try:
                img_pil = Image.open(path).convert('RGB')
                embedding = self.face_processor.get_embedding(img_pil)
                if embedding is not None:
                    processed_data.append({'path': path, 'embedding': embedding})
                else:
                    QMessageBox.warning(self, "Error", f"No face detected in {path}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error Processing Image", str(e))
                return

        # Save to DB
        try:
            # Check for duplicates by name
            existing_identities = self.db.get_all_identities_with_embeddings()
            duplicate = None
            for ident in existing_identities:
                if ident['name'].lower() == name.lower() and (not self.identity_data or ident['id'] != self.identity_data['id']):
                    duplicate = ident
                    break
                    
            if duplicate:
                reply = QMessageBox.question(self, "Duplicate Name", f"An identity named '{name}' already exists. Do you want to merge this profile into the existing one?", 
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    target_category = duplicate['category']
                    if target_category != category:
                        cat_reply = QMessageBox.question(self, "Category Conflict", f"The existing profile is '{target_category}', but you selected '{category}'.\nClick YES to keep '{target_category}', or NO to change it to '{category}'.", 
                                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        if cat_reply == QMessageBox.StandardButton.No:
                            target_category = category
                    
                    # Perform Merge
                    if target_category != duplicate['category'] or duplicate['name'] != name:
                        self.db.update_identity(duplicate['id'], name, target_category)
                    
                    # If we were editing an existing profile, merge its photos into the duplicate
                    if self.identity_data:
                        self.db.merge_identities(self.identity_data['id'], duplicate['id'])
                    
                    for item in processed_data:
                        self.db.add_embedding(duplicate['id'], item['path'], item['embedding'])
                        
                    QMessageBox.information(self, "Success", "Profiles merged successfully.")
                    self.accept()
                    return
                else:
                    return # Allow them to rethink name without saving
                    
            if self.identity_data:
                # Update existing identity name and category
                self.db.update_identity(self.identity_data['id'], name, category)
                
                # Add any new embeddings
                for item in processed_data:
                    self.db.add_embedding(self.identity_data['id'], item['path'], item['embedding'])
                    
                QMessageBox.information(self, "Success", "Identity updated successfully.")
            else:
                self.db.add_identity(name, category, processed_data)
                QMessageBox.information(self, "Success", f"Identity {name} added successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Database Error", str(e))


class IdentityTab(QWidget):
    """ Tab showing identities in a neat grid based on category """
    def __init__(self, parent_main_window, filter_categories=None):
        super().__init__()
        self.main_app = parent_main_window
        self.db = self.main_app.db
        self.filter_categories = filter_categories if filter_categories else ["VIP", "Blacklist"]
        
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        title_text = "Registered Identities" if "VIP" in self.filter_categories else "Unknown Personalities"
        toolbar.addWidget(QLabel(f"<h2>{title_text}</h2>"))
        toolbar.addStretch()
        
        if "VIP" in self.filter_categories:
            self.btn_batch = QPushButton("Process Video")
            self.btn_batch.setStyleSheet("background-color: #f9e2af; color: #11111b; font-weight: bold; padding: 8px 15px; border-radius: 5px;")
            self.btn_batch.clicked.connect(self.open_video_processor)
            toolbar.addWidget(self.btn_batch)
            
            self.btn_add = QPushButton("Enrol New Person")
            self.btn_add.setObjectName("secondary")
            self.btn_add.clicked.connect(self.open_add_dialog)
            toolbar.addWidget(self.btn_add)
        
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.populate_grid)
        toolbar.addWidget(self.btn_refresh)
        
        layout.addLayout(toolbar)
        
        # Grid Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

    def populate_grid(self):
        # Clear grid
        for i in reversed(range(self.grid.count())): 
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                
        identities = self.db.get_all_identities_with_embeddings()
        
        col = 0
        row = 0
        max_cols = max(1, self.scroll.width() // 250)
        
        for identity in identities:
            if identity['category'] not in self.filter_categories:
                continue
            card = self.create_card(identity)
            self.grid.addWidget(card, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        self.grid.setRowStretch(row + 1, 1) # Push everything up

    def create_card(self, identity):
        frame = QFrame()
        frame.setStyleSheet("background-color: #313244; border-radius: 8px;")
        frame.setFixedSize(220, 260)
        
        layout = QVBoxLayout(frame)
        
        # Thumbnail (use first embedding if available)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if identity['embeddings'] and len(identity['embeddings']) > 0:
            path = identity['embeddings'][0]['image_path']
            try:
                pixmap = QPixmap(path).scaled(180, 150, Qt.AspectRatioMode.KeepAspectRatio)
                img_label.setPixmap(pixmap)
            except:
                img_label.setText("Image Error")
        else:
            img_label.setText("No Images")
        
        layout.addWidget(img_label)
        
        name_lbl = QLabel(f"<b>{identity['name']}</b>")
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_lbl)
        
        cat_color = "#a6e3a1" if identity['category'] == "VIP" else "#f38ba8"
        cat_lbl = QLabel(f"<span style='color:{cat_color}'>{identity['category']}</span> | Photos: {len(identity['embeddings'])}")
        cat_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(cat_lbl)
        
        btn_layout = QHBoxLayout()
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(lambda _, id_data=identity: self.open_edit_dialog(id_data))
        
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(lambda _, iid=identity['id']: self.delete_identity(iid))
        
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(del_btn)
        layout.addLayout(btn_layout)
        
        return frame

    def open_add_dialog(self):
        dlg = AddEditIdentityDialog(self.db, self.main_app.face_processor, parent=self)
        if dlg.exec():
            self.main_app.refresh_all_grids()

    def open_edit_dialog(self, identity_data):
        dlg = AddEditIdentityDialog(self.db, self.main_app.face_processor, identity_data=identity_data, parent=self)
        if dlg.exec():
            self.main_app.refresh_all_grids()

    def delete_identity(self, iid):
        confirm = QMessageBox.question(self, "Confirm Delete", "Delete this entire person and all their photos from the DB?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.db.delete_identity(iid)
            self.main_app.refresh_all_grids()
            
    def open_video_processor(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            from PyQt6.QtWidgets import QProgressDialog
            dlg = VideoProcessorDialog(file_path, self.main_app.face_processor, self.db, self)
            dlg.exec()
            self.main_app.refresh_all_grids()

import os
import math
from PyQt6.QtWidgets import QProgressBar

def compute_tracking_score(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    # Centroid distance limits fragmentation from rapid head movement
    cxA, cyA = (boxA[0]+boxA[2])/2, (boxA[1]+boxA[3])/2
    cxB, cyB = (boxB[0]+boxB[2])/2, (boxB[1]+boxB[3])/2
    dist = math.hypot(cxA - cxB, cyA - cyB)
    
    width = max((boxA[2] - boxA[0]), (boxB[2] - boxB[0]))
    normalized_dist = dist / (width + 1e-6)
    
    score = iou + max(0, 1.0 - normalized_dist)
    return score

class VideoProcessorThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, video_path, face_processor, db):
        super().__init__()
        self.video_path = video_path
        self.face_processor = face_processor
        self.db = db
        self.save_dir = os.path.join(os.getcwd(), "captured_faces")
        os.makedirs(self.save_dir, exist_ok=True)
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30
            
        # We process frequently to ensure robust intersection-over-union tracking.
        frame_interval = max(1, fps // 5) # Process roughly at 5 FPS
        
        active_tracks = [] # list of dicts: {'id': int, 'last_box': bbox, 'embeddings': [], 'images': [], 'misses': 0}
        finished_tracks = []
        next_track_id = 1
        
        frame_idx = 0
        added_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                boxes, embeddings = self.face_processor.extract_faces_and_embeddings(frame)
                
                unmatched_boxes = set(range(len(boxes)))
                
                    # Match active tracks to new boxes via Score
                for track in active_tracks:
                    track['misses'] += 1
                    
                    best_iou = 0
                    best_box_idx = -1
                    for i in unmatched_boxes:
                        score = compute_tracking_score(track['last_box'], boxes[i])
                        if score > best_iou:
                            best_iou = score
                            best_box_idx = i
                    
                    if best_iou > 0.4: # Minimum overlap/distance required
                        box = boxes[best_box_idx]
                        track['last_box'] = box
                        track['embeddings'].append(embeddings[best_box_idx])
                        
                        # Crop safe boundaries
                        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(frame.shape[1], int(box[2])), min(frame.shape[0], int(box[3]))
                        if x2 - x1 >= 10 and y2 - y1 >= 10:
                            track['images'].append(frame[y1:y2, x1:x2])
                            
                        track['misses'] = 0
                        unmatched_boxes.remove(best_box_idx)
                        
                # New tracks for remaining boxes
                for i in unmatched_boxes:
                    box = boxes[i]
                    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(frame.shape[1], int(box[2])), min(frame.shape[0], int(box[3]))
                    if x2 - x1 >= 10 and y2 - y1 >= 10:
                        active_tracks.append({
                            'id': next_track_id,
                            'last_box': box,
                            'embeddings': [embeddings[i]],
                            'images': [frame[y1:y2, x1:x2]],
                            'misses': 0
                        })
                        next_track_id += 1
                        
                # Remove stale tracks
                still_active = []
                for track in active_tracks:
                    if track['misses'] > 5:
                        finished_tracks.append(track)
                    else:
                        still_active.append(track)
                active_tracks = still_active
                
            frame_idx += 1
            if frame_idx % fps == 0 and total_frames > 0:
                 percent = int((frame_idx / total_frames) * 100)
                 self.progress.emit(min(percent, 90)) # Reserve last 10% for DB finalization
                 
        cap.release()
        
        # Merge remaining active tracks to finished
        finished_tracks.extend(active_tracks)
        
        self.log.emit(f"Tracking concluded. Aggregating {len(finished_tracks)} tracks into Database...")
        known_identities = self.db.get_all_identities_with_embeddings()
        unmatched_tracks = []
        
        for track in finished_tracks:
            # Drop noisy/false-positive tracks that appeared very briefly
            if len(track['embeddings']) < 3:
                continue
                
            best_match_id = None
            best_match_name = None
            best_sim_global = -1.0
            
            # Subsample track images so we avoid bloating DB with identical 1-second interval picks
            total_frames_in_track = len(track['images'])
            keep_count = min(5, total_frames_in_track)
            indices_to_save = np.linspace(0, total_frames_in_track - 1, keep_count, dtype=int)
            
            # Find closest matching DB identity using Track-level global best similarity
            for db_ident in known_identities:
                for db_emb_dict in db_ident['embeddings']:
                    db_emb = db_emb_dict['embedding']
                    db_emb_norm = db_emb / (np.linalg.norm(db_emb) + 1e-8)
                    
                    for trk_emb in track['embeddings']:
                        trk_emb_norm = trk_emb / (np.linalg.norm(trk_emb) + 1e-8)
                        sim = np.dot(trk_emb_norm, db_emb_norm)
                        
                        if sim > best_sim_global:
                            best_sim_global = sim
                            if sim > 0.75: # Update to a stricter threshold so different people aren't clubbed
                                best_match_id = db_ident['id']
                                best_match_name = db_ident['name']
                                
            if best_match_id is not None:
                for idx in indices_to_save:
                    filename = f"track_{track['id']}_{idx}_{np.random.randint(10000)}.jpg"
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, track['images'][idx])
                    self.db.add_embedding(best_match_id, filepath, track['embeddings'][idx])
                    added_count += 1
                self.log.emit(f"Track {track['id']} -> Grouped into: {best_match_name} ({keep_count} photos)")
            else:
                unmatched_tracks.append({'track': track, 'indices': indices_to_save})
                
        # 8. Unknown Identity Handling: Within-video clustering!
        # Group remaining unmatched tracks together
        grouped_unmatched = [] # lists of unmatched track dicts
        for ut_dict in unmatched_tracks:
            track = ut_dict['track']
            merged = False
            for group in grouped_unmatched:
                group_sim = -1.0
                for g_ut_dict in group:
                    g_track = g_ut_dict['track']
                    for e1 in track['embeddings']:
                        e1_n = e1 / (np.linalg.norm(e1) + 1e-8)
                        for e2 in g_track['embeddings']:
                            e2_n = e2 / (np.linalg.norm(e2) + 1e-8)
                            sim = np.dot(e1_n, e2_n)
                            if sim > group_sim:
                                group_sim = sim
                if group_sim > 0.72: # Update to a sterner threshold for distinct unknown person tracks
                    group.append(ut_dict)
                    merged = True
                    break
            
            if not merged:
                grouped_unmatched.append([ut_dict])
                
        # Commit clustered unknown tracks
        for group in grouped_unmatched:
            rand_id = f"Unknown_Group_{np.random.randint(1000, 9999)}"
            images_data = []
            for ut_dict in group:
                track = ut_dict['track']
                indices_to_save = ut_dict['indices']
                for idx in indices_to_save:
                    filename = f"track_{track['id']}_{idx}_{np.random.randint(10000)}.jpg"
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, track['images'][idx])
                    images_data.append({'path': filepath, 'embedding': track['embeddings'][idx]})
            
            if images_data:
                self.db.add_identity(rand_id, 'Unknown', images_data)
                added_count += len(images_data)
            self.log.emit(f"Clustered {len(group)} Tracks -> Generated new Unknown: {rand_id} ({len(images_data)} photos)")
            
        self.progress.emit(100)
        self.log.emit(f"Batch Processing Complete! Saved {added_count} facial embeddings.")
        self.finished.emit()

class VideoProcessorDialog(QDialog):
    def __init__(self, video_path, face_processor, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Video...")
        self.setFixedSize(500, 300)
        
        layout = QVBoxLayout(self)
        self.lbl_info = QLabel(f"Processing: {os.path.basename(video_path)}")
        layout.addWidget(self.lbl_info)
        
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        
        self.log_widget = QListWidget()
        layout.addWidget(self.log_widget)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        self.btn_close.setEnabled(False)
        layout.addWidget(self.btn_close)
        
        self.thread = VideoProcessorThread(video_path, face_processor, db)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.log.connect(self.log_widget.addItem)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()
        
    def on_finished(self):
        self.btn_close.setEnabled(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition CCTV Pro")
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet(DARK_THEME_QSS)
        
        self.db = DatabaseManager('face_db.sqlite')
        self.face_processor = FaceProcessor(self.db)
        
        self.init_ui()
        
        # Stop CV backend until Live Monitor tab opens
        self.video_thread = None

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Tab 1: Live Monitor
        self.live_tab = QWidget()
        live_layout = QVBoxLayout(self.live_tab)
        
        self.video_label = QLabel("Camera Offline. Click Start Stream.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px; font-size: 18px;")
        live_layout.addWidget(self.video_label, stretch=1)
        
        self.btn_toggle_cam = QPushButton("Start Stream")
        self.btn_toggle_cam.setObjectName("success")
        self.btn_toggle_cam.clicked.connect(self.toggle_camera)
        live_layout.addWidget(self.btn_toggle_cam, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.tabs.addTab(self.live_tab, "Live Monitor")
        
        # Tab 2: Identity Management
        self.identities_tab = IdentityTab(self, filter_categories=["VIP", "Blacklist"])
        self.tabs.addTab(self.identities_tab, "Identities & Configuration")
        
        # Tab 3: Unknown Personalities
        self.unknowns_tab = IdentityTab(self, filter_categories=["Unknown"])
        self.tabs.addTab(self.unknowns_tab, "Unknown Personalities")

        # Detection Logs Tab
        self.log_tab = QWidget()
        self.log_layout = QVBoxLayout(self.log_tab)
        
        log_filter_layout = QHBoxLayout()
        self.log_search_input = QLineEdit()
        self.log_search_input.setPlaceholderText("Search detections by Name...")
        self.log_search_input.textChanged.connect(self.refresh_logs)
        
        self.log_category_combo = QComboBox()
        self.log_category_combo.addItems(["All", "VIP", "Blacklist"])
        self.log_category_combo.currentTextChanged.connect(self.refresh_logs)
        
        log_filter_layout.addWidget(QLabel("Search Logs:"))
        log_filter_layout.addWidget(self.log_search_input)
        log_filter_layout.addWidget(QLabel("Category Filter:"))
        log_filter_layout.addWidget(self.log_category_combo)
        
        self.log_layout.addLayout(log_filter_layout)
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["Name", "Category", "Timestamp"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_layout.addWidget(self.log_table)
        
        btn_layout = QHBoxLayout()
        self.btn_del_log = QPushButton("Delete Selected Log")
        self.btn_del_log.clicked.connect(self.delete_selected_log)
        self.btn_clear_logs = QPushButton("Clear All Logs")
        self.btn_clear_logs.clicked.connect(self.clear_all_logs)
        btn_layout.addWidget(self.btn_del_log)
        btn_layout.addWidget(self.btn_clear_logs)
        self.log_layout.addLayout(btn_layout)
        
        self.tabs.addTab(self.log_tab, "Detection Logs")
        
        self.refresh_logs()
        
        # When tab changes, handle grid refresh or camera pause
        self.tabs.currentChanged.connect(self.on_tab_change)

    def refresh_logs(self):
        category = self.log_category_combo.currentText()
        search = self.log_search_input.text().strip()
        
        logs = self.db.get_detection_logs(category_filter=category, name_search=search)
        
        self.log_table.setRowCount(0)
        for i, log in enumerate(logs):
            self.log_table.insertRow(i)
            item_name = QTableWidgetItem(log['name'])
            item_name.setData(Qt.ItemDataRole.UserRole, log['id'])
            self.log_table.setItem(i, 0, item_name)
            self.log_table.setItem(i, 1, QTableWidgetItem(log['category']))
            self.log_table.setItem(i, 2, QTableWidgetItem(log['timestamp']))

    def delete_selected_log(self):
        row = self.log_table.currentRow()
        if row < 0:
            return
            
        log_id = self.log_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(self, 'Confirm Delete', 'Are you sure you want to delete this log entry?', 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_detection_log(log_id)
            self.refresh_logs()
            
    def clear_all_logs(self):
        reply = QMessageBox.question(self, 'Confirm Clear', 'Are you sure you want to delete ALL detection logs?', 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.db.clear_all_detection_logs()
            self.refresh_logs()

    def refresh_all_grids(self):
        self.identities_tab.populate_grid()
        self.unknowns_tab.populate_grid()
        self.refresh_logs()

    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
            self.btn_toggle_cam.setText("Start Stream")
            self.btn_toggle_cam.setObjectName("success")
            self.btn_toggle_cam.setStyleSheet("") # trick to reappply qss
            self.video_label.clear()
            self.video_label.setText("Camera Offline.")
        else:
            self.video_thread = FaceVideoThread(self.face_processor, self.db)
            self.video_thread.frame_ready.connect(self.update_image)
            self.video_thread.log_updated.connect(self.refresh_logs)
            self.video_thread.start()
            self.btn_toggle_cam.setText("Stop Stream")
            self.btn_toggle_cam.setObjectName("danger")
            self.btn_toggle_cam.setStyleSheet("")

    def update_image(self, cv_img, display_data):
        frame = cv2.flip(cv_img, 1) # Flip image first so text renders correctly
        h, w, ch = frame.shape
        
        for res in display_data:
            box = res['box']
            name = res['name']
            category = res['category']
            
            # mirror coordinate box bounds dynamically!
            # mtcnn boxes are stored as [x1, y1, x2, y2]
            x1 = w - box[2]
            x2 = w - box[0]
            # y coordinates don't mirror vertically
            y1 = box[1]
            y2 = box[3]
            
            if category == 'VIP':
                color = (0, 255, 0)
            elif category == 'Blacklist':
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
                
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label_text = f"{name} ({category})"
            cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def on_tab_change(self, index):
        if index == 1:
            self.identities_tab.populate_grid()
        elif index == 2:
            self.unknowns_tab.populate_grid()

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
