import sys
import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QComboBox, QFileDialog, QListWidget, QDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

from db_manager import DatabaseManager
from face_pipeline import FaceProcessor

class VideoThread(QThread):
    # Emit processed frame with annotations
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, face_processor):
        super().__init__()
        self.face_processor = face_processor
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # Process frame
                results = self.face_processor.process_frame(frame)
                
                # Annotate frame
                for res in results:
                    box = res['box']
                    name = res['name']
                    category = res['category']
                    similarity = res['similarity']
                    
                    # Set color based on category
                    if category == 'VIP':
                        color = (0, 255, 0) # Green (BGR in OpenCV)
                    elif category == 'Blacklist':
                        color = (0, 0, 255) # Red (BGR in OpenCV)
                    else:
                        color = (128, 128, 128) # Gray (BGR)
                        
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    label_text = f"{name} ({category}) - {similarity:.2f}" if name != "Unknown" else "Unknown"
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class AddIdentityDialog(QDialog):
    def __init__(self, db_manager, face_processor, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.face_processor = face_processor
        self.setWindowTitle("Add Identity")
        self.setFixedSize(400, 350)
        self.img_path = None
        
        layout = QVBoxLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")
        layout.addWidget(self.name_input)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["VIP", "Blacklist"])
        layout.addWidget(self.category_combo)
        
        self.image_label = QLabel("No Image Selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #313244;")
        self.image_label.setFixedSize(150, 150)
        
        # Center the image label
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.image_label)
        img_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(img_layout)
        
        btn_layout = QHBoxLayout()
        self.btn_select_img = QPushButton("Select Image")
        self.btn_select_img.clicked.connect(self.select_image)
        btn_layout.addWidget(self.btn_select_img)
        
        self.btn_save = QPushButton("Save Identity")
        self.btn_save.clicked.connect(self.save_identity)
        self.btn_save.setStyleSheet("background-color: #a6e3a1; color: #11111b;")
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.img_path = file_name
            pixmap = QPixmap(self.img_path).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def save_identity(self):
        name = self.name_input.text().strip()
        category = self.category_combo.currentText()
        
        if not name or not self.img_path:
            QMessageBox.warning(self, "Error", "Please provide a name and select an image.")
            return
            
        try:
            # Load image and get embedding
            img_pil = Image.open(self.img_path).convert('RGB')
            embedding = self.face_processor.get_embedding(img_pil)
            
            if embedding is None:
                QMessageBox.warning(self, "Error", "No face detected in the image.")
                return
                
            self.db.add_identity(name, category, self.img_path, embedding)
            QMessageBox.information(self, "Success", f"Identity {name} added successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition CCTV")
        self.setGeometry(100, 100, 1000, 700)
        
        self.db = DatabaseManager('face_db.sqlite')
        self.face_processor = FaceProcessor(self.db)
        
        self.init_ui()
        
        self.thread = VideoThread(self.face_processor)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def init_ui(self):
        # Apply dark theme using Catppuccin Mocha colors
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; }
            QPushButton { 
                background-color: #89b4fa; color: #11111b; 
                border-radius: 5px; padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
            QListWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #313244; font-size: 14px; }
            QLineEdit, QComboBox { 
                background-color: #181825; color: #cdd6f4; border: 1px solid #313244; 
                padding: 5px; border-radius: 3px; 
            }
        """)
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left side - Video Feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, stretch=2)
        
        # Right side - Controls and Lists
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        title = QLabel("CCTV Monitor")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(title)
        
        self.btn_add_identity = QPushButton("Add New Identity")
        self.btn_add_identity.setStyleSheet("background-color: #cba6f7; color: #11111b; border-radius: 5px; padding: 10px; font-weight: bold;")
        self.btn_add_identity.clicked.connect(self.open_add_dialog)
        right_panel.addWidget(self.btn_add_identity)
        
        list_label = QLabel("Current Identities:")
        font_label = QFont()
        font_label.setPointSize(12)
        font_label.setBold(True)
        list_label.setFont(font_label)
        right_panel.addWidget(list_label)
        
        self.list_identities = QListWidget()
        right_panel.addWidget(self.list_identities)
        
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_list)
        right_panel.addWidget(self.btn_refresh)
        
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_identity)
        self.btn_delete.setStyleSheet("background-color: #f38ba8; color: #11111b; border-radius: 5px; padding: 10px; font-weight: bold;")
        right_panel.addWidget(self.btn_delete)
        
        main_layout.addLayout(right_panel, stretch=1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.refresh_list()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
        
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def open_add_dialog(self):
        dlg = AddIdentityDialog(self.db, self.face_processor, self)
        if dlg.exec():
            self.refresh_list()

    def refresh_list(self):
        self.list_identities.clear()
        identities = self.db.get_all_identities()
        for idx in identities:
            display_text = f"[{idx['category']}] {idx['name']} (ID: {idx['id']})"
            self.list_identities.addItem(display_text)

    def delete_identity(self):
        selected = self.list_identities.currentItem()
        if selected:
            text = selected.text()
            # Extract ID from text e.g., "[VIP] John (ID: 1)"
            import re
            match = re.search(r'\(ID: (\d+)\)', text)
            if match:
                identity_id = int(match.group(1))
                self.db.delete_identity(identity_id)
                self.refresh_list()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
