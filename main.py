import sys
import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QFileDialog, QListWidget, QDialog, QMessageBox,
    QTabWidget, QScrollArea, QFrame, QSizePolicy
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

# --- Threads ---
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
    """ Tab showing all identities in a neat grid """
    def __init__(self, parent_main_window):
        super().__init__()
        self.main_app = parent_main_window
        self.db = self.main_app.db
        
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("<h2>Registered Identities</h2>"))
        toolbar.addStretch()
        
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
        # Determine number of columns based on width
        max_cols = max(1, self.scroll.width() // 250)
        
        for identity in identities:
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
            self.populate_grid()

    def open_edit_dialog(self, identity_data):
        dlg = AddEditIdentityDialog(self.db, self.main_app.face_processor, identity_data=identity_data, parent=self)
        if dlg.exec():
            self.populate_grid()

    def delete_identity(self, iid):
        confirm = QMessageBox.question(self, "Confirm Delete", "Delete this entire person and all their photos from the DB?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.db.delete_identity(iid)
            self.populate_grid()

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
        self.thread = None

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
        self.identities_tab = IdentityTab(self)
        self.tabs.addTab(self.identities_tab, "Identities & Configuration")
        
        # When tab changes, handle grid refresh or camera pause
        self.tabs.currentChanged.connect(self.on_tab_change)

    def toggle_camera(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
            self.btn_toggle_cam.setText("Start Stream")
            self.btn_toggle_cam.setObjectName("success")
            self.btn_toggle_cam.setStyleSheet("") # trick to reappply qss
            self.video_label.clear()
            self.video_label.setText("Camera Offline.")
        else:
            self.thread = VideoThread(self.face_processor)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
            self.btn_toggle_cam.setText("Stop Stream")
            self.btn_toggle_cam.setObjectName("danger")
            self.btn_toggle_cam.setStyleSheet("")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def on_tab_change(self, index):
        if index == 1:
            self.identities_tab.populate_grid()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
