# Face Recognition CCTV Pro

A modern, highly accurate, desktop-based Face Recognition CCTV system built with **PyQt6**, **OpenCV**, and **PyTorch**. This application transforms any standard webcam into a sophisticated local security monitor capable of instant VIP and Blacklist recognition.

![Face Recognition CCTV Pro](https://raw.githubusercontent.com/pytorch/vision/main/docs/source/_static/img/pytorch-logo-dark.png) *(Note: Add actual app screenshots here)*

## 🌟 Key Features

- **Live GPU/MPS-Accelerated Detection**: Utilizes MTCNN to detect faces in real-time, leveraging NVIDIA CUDA or Apple Silicon MPS chips for ultra-fast, smooth video inference.
- **Deep Learning Embeddings**: Leverages the robust `InceptionResnetV1` (trained on VGGFace2) to generate 512-dimensional facial embeddings.
- **Multi-Photo Enrollment Engine**: Users can be registered with multiple reference photos from different angles (left profile, right profile, varying lighting). The matching algorithm aggregates similarities to guarantee unparalleled real-world accuracy.
- **Integrated SQLite Storage**: A fully-offline, lightweight relational database manages profiles, eliminating the need for complex database servers.
- **Premium User Interface**: A beautifully crafted PyQt6 dashboard featuring a dark Catppuccin theme, Grid profile views, Tabbed navigation, and live webcam snapshot workflows.
- **Fully Standalone Deployment**: Can be packaged into a single Windows/Mac Executable (.exe / .app) where end users don't need Python installed.

---

## 🏗️ System Architecture

The application is built around three tightly coupled, yet modular components:

### 1. The Presentation Layer (PyQt6 `main.py`)
This is the command center. We use `QTabWidget` to split the UX into two domains:
- **Live Monitor**: Uses PyQt `QThread` to spawn a dedicated worker thread for the Webcam. This ensures the heavy ML model execution doesn't block or freeze the UI thread. OpenCV matrices are seamlessly converted into PyQt QPixmaps for 30FPS UI rendering.
- **Identity Manager**: A robust Grid-Layout interface to view, edit, and delete enrolled individuals. Contains sub-dialogs for multi-file uploading and PyQt QTimers for live-capturing enrollment photos directly from the camera.

### 2. The Storage Layer (`db_manager.py`)
Built on built-in `sqlite3`.
- Handles complex 1-to-many schema tracking identities and their accompanying Numpy Array embeddings over `BLOB` formats. Custom adapters and converters automatically marshal PyTorch tensors to Numpy arrays to raw bytes for SQL commits in real time.

### 3. The ML Inference Pipeline (`face_pipeline.py`)
Built strictly with PyTorch and `facenet-pytorch`.
1. **Pre-Processing**: The image stream is captured via OpenCV (BGR) and converted to PIL format (RGB) for inference.
2. **MTCNN (Multi-task Cascaded Convolutional Networks)**: Extracts bounding boxes around detected faces in the frame.
3. **InceptionResnetV1**: The cropped faces are fed through the ResNet architecture. This outputs a 512-dimensional vector.
4. **Distance Calculation**: The live vector is mathematically compared (Cosine Similarity) against the matrix of enrolled vectors in the Database. The highest probability match dictates the label (VIP / Blacklisted / Unknown).

---

## 🚀 Installation & Setup (For Developers)

We provide an OS-aware installation script that automatically detects your operating system and installs the correct `PyQt6` and `PyTorch` wheels depending on if you are on Windows, macOS, or Linux.

### Prerequisites
- **Python 3.9 - 3.11** installed.
- A functional Web Camera.
- *(Optional but Recommended)*: NVIDIA GPU with CUDA drivers OR an Apple Silicon Mac (M1/M2/M3) for hardware acceleration.

### Quick Start
1. Clone or download the repository.
2. Open a terminal / command prompt in the project root.
3. Create and activate a Virtual Environment (highly recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS and Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Run the automated installer script:
   ```bash
   python install.py
   ```
5. Launch the application!
   ```bash
   python main.py
   ```

*(On the very first execution, the pipeline will seamlessly download approx. 100MB of pre-trained VGGFace2 weights).*

---

## 📦 Distribution (Standalone Executable)

If you are a developer looking to distribute this software to non-technical users, you can compile the entire source code, including Python, OpenCV, and the heavyweight PyTorch ML models, into a single distributable folder.

1. Ensure PyInstaller is installed (`pip install pyinstaller`).
2. We have provided a configured `build.spec` file and a `hook-facenet_pytorch.py` file to handle the complex model dynamic linking.
3. Compile the app by running:
   ```bash
   pyinstaller build.spec
   ```
4. Once completed, navigate to the newly created `dist/` directory. 
5. You can **Zip** the entire `FaceRecognitionCCTV` subfolder located inside `dist/` and send it to anyone! They simply extract it and double-click the executable inside to run the system natively.
