# Face Recognition CCTV Pro

A modern, highly accurate, desktop-based Face Recognition CCTV system built with **PyQt6**, **OpenCV**, and **PyTorch**. 

## Features
- live GPU/MPS-accelerated face detection using MTCNN.
- High accuracy facial embeddings via `InceptionResnetV1` (VGGFace2).
- Robust local data storage using SQLite for saving user profiles and reference embeddings.
- Multi-image enrollment: Users can be enrolled with multiple photos from varying angles for maximum real-world reliability.
- Live webcam integration with bounding boxes and matching scores.
- Platform independence: Supports Windows (CUDA/CPU), macOS (Apple Silicon MPS/CPU), and Linux natively.

## Prerequisites
- **Python 3.9 - 3.11** installed.
- A functional Web Camera (default index 0 is used).
- *(Optional but Recommended)*: NVIDIA GPU with CUDA Toolkit installed, OR an Apple Silicon Mac (M1/M2/M3) for hardware acceleration.

## Installation (Cross-Platform)

We provide an OS-aware installation script that automatically detects your operating system and installs the correct `PyQt6` and `PyTorch` wheels (handling known DLL issues on Windows natively while keeping macOS/Linux clean).

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
4. Run the installer script:
   ```bash
   python install.py
   ```

*(On the very first execution, the pipeline will seamlessly download approx. 100MB of pre-trained VGGFace2 weights).*

## Usage

Start the application by running:
```bash
python main.py
```

### 1. Enrollment
- Navigate to the **"Identities & Configuration"** tab in the sidebar/tabs.
- Click **"Enroll New Person"**.
- Enter a name and select a category (VIP or Blacklist).
- Use **Upload Files** to select existing photos, or **Take Photo via Camera** to take live snapshots. *(Providing 3-5 photos from different angles/lighting conditions drastically improves accuracy)*.
- Click **"Save & Enroll"**.

### 2. Live Monitor
- Navigate to the **"Live Monitor"** tab.
- Click **"Start Stream"**.
- Step in front of the camera to see real-time detection and verification.
  - 🟩 **Green Box**: Verified VIP Match.
  - 🟥 **Red Box**: Verified Blacklist Match.
  - ⬜ **Gray Box**: Unknown person.
