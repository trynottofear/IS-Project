# 👁️ Face Recognition CCTV Pro

A production-grade, fully offline, desktop Face Recognition & CCTV surveillance system built on **PyTorch**, **OpenCV**, and **PyQt6**. It transforms any standard webcam or pre-recorded video file into an intelligent security monitor capable of real-time VIP/Blacklist identification, automated unknown-face harvesting, temporal tracking, and robust identity clustering — all running locally with zero cloud dependency.

---

## 📚 Table of Contents

1.  [Core Features at a Glance](#-core-features-at-a-glance)
2.  [Codebase Structure (Every File Explained)](#-codebase-structure)
3.  [Deep-Learning Pipeline (Models & Algorithms)](#-deep-learning-pipeline)
4.  [Face Tracking Algorithm (IoU + Centroid)](#-face-tracking-algorithm)
5.  [Identity Locking (Temporal Voting System)](#-identity-locking-temporal-voting)
6.  [Post-Processing: Matching & Clustering Engine](#-post-processing-matching--clustering)
7.  [Detection Log Cooldown Mechanism](#-detection-log-cooldown)
8.  [User Interface (Every Screen & Feature)](#-user-interface)
9.  [Database Architecture](#-database-architecture)
10. [Installation & Setup](#-installation--setup)
11. [Building the Standalone Desktop App (.exe)](#-building-the-standalone-desktop-app)

---

## 🌟 Core Features at a Glance

| Feature | Details |
|---|---|
| **Real-Time GPU Detection** | MTCNN face detection accelerated via NVIDIA CUDA or Apple MPS, rendering at ~30 FPS |
| **512-D Deep Embeddings** | InceptionResnetV1 (VGGFace2) generates dense facial feature vectors |
| **Temporal Voting** | 30-frame rolling window prevents flickering misidentifications in live streams |
| **Auto-Clustering** | Unknown faces are automatically grouped by visual similarity and stored as new profiles |
| **Multi-Photo Enrollment** | Register people with unlimited reference photos (file upload or live webcam snapshots) |
| **Merge & Edit** | Reclassify unknowns into VIP/Blacklist, merge duplicate profiles, delete individual photos |
| **10-Min Log Cooldown** | Prevents detection log spam when a person stands idle in front of the camera |
| **Standalone .exe Build** | Ship the entire app (Python + PyTorch + OpenCV) as a single-click Windows executable |
| **Dark Catppuccin Theme** | Premium dark UI designed for extended monitoring sessions |

---

## 📁 Codebase Structure

```
face_recognition_app/
├── main.py                          # Application entry point & all UI/thread logic
├── face_pipeline.py                 # ML inference engine (MTCNN + InceptionResnetV1)
├── db_manager.py                    # SQLite database abstraction layer
├── install.py                       # Cross-platform automated dependency installer
├── requirements.txt                 # Pinned dependency versions
├── build.spec                       # PyInstaller compilation specification
├── fix_db.py                        # One-time database foreign-key repair utility
├── hooks/
│   └── hook-facenet_pytorch.py      # PyInstaller hook for facenet model weights
├── captured_faces/                  # Auto-generated folder for cropped face images
├── face_db.sqlite                   # Auto-generated SQLite database file
└── venv/                            # Python virtual environment (not committed)
```

### File-by-File Breakdown

#### `main.py` (~1500 lines)
The nerve center.  Contains **everything** related to the graphical interface and the background processing threads:

| Class / Function | Role |
|---|---|
| `DARK_THEME_QSS` | Global Catppuccin Macchiato dark stylesheet applied to every widget |
| `FaceVideoThread` | QThread for **Detection Only** mode — reads camera, runs inference, draws boxes, logs VIP/Blacklist entries with 10-min cooldown |
| `VideoThread` | Legacy simple camera preview thread (used internally) |
| `CaptureDialog` | Pop-up dialog that opens the webcam and lets the user take a snapshot photo for enrollment |
| `AddEditIdentityDialog` | Full identity management dialog — name input, category selector, file upload, camera capture, photo gallery with per-photo delete, duplicate detection with merge prompt |
| `IdentityTab` | Reusable grid-card widget that displays either Registered Identities or Unknown Personalities depending on `filter_categories` |
| `compute_tracking_score()` | Pure function computing IoU + normalized centroid distance for frame-to-frame face tracking |
| `VideoProcessorThread` | QThread for **Batch Processing** mode — processes an entire video file or camera stream silently, then clusters and saves all faces to DB |
| `VideoProcessorDialog` | Progress dialog for batch processing — shows live camera preview when processing from webcam |
| `HybridCaptureThread` | QThread for **Detection + Capture** mode — combines live bounding-box display with silent background face harvesting and post-capture clustering |
| `MainWindow` | The top-level application window — creates all three tabs, manages thread lifecycle, handles tab switching |

#### `face_pipeline.py` (~128 lines)
Isolated ML inference module.  Zero UI code — purely mathematical:

| Component | Details |
|---|---|
| `FaceProcessor.__init__()` | Auto-detects hardware: tries `cuda:0` → `mps` → `cpu`. Loads MTCNN (multi-face) + MTCNN (single-face for enrollment) + InceptionResnetV1 with `vggface2` pretrained weights |
| `get_embedding(img_pil)` | Takes a single PIL image → detects one face → returns its 512-D numpy embedding. Used during manual enrollment |
| `extract_faces_and_embeddings(frame_cv)` | Takes an OpenCV BGR frame → returns raw bounding boxes + embeddings without any DB matching. Used by `VideoProcessorThread` for batch processing |
| `process_frame(frame_cv)` | The full pipeline: detect faces → extract embeddings → normalize → compare against **every** enrolled embedding in DB via cosine similarity → return labeled results with `name`, `category`, `identity_id`, `similarity`, `embedding` per face |

#### `db_manager.py` (~348 lines)
All database operations.  Custom numpy serialization:

| Method | What It Does |
|---|---|
| `adapt_array()` / `convert_array()` | Registered as SQLite adapter/converter — transparently serializes `np.ndarray` ↔ binary `BLOB` using `np.save()`/`np.load()` over `io.BytesIO` streams |
| `init_db()` | Creates tables if missing. Runs schema migration checks (V1→V2→V3) and self-healing foreign-key repair |
| `add_identity()` | Inserts a new person with name + category + list of `{path, embedding}` dicts |
| `add_embedding()` | Appends an additional reference photo/embedding to an existing identity |
| `update_identity()` | Changes name and/or category (e.g., Unknown → VIP) |
| `delete_identity()` | Removes a person — `ON DELETE CASCADE` automatically purges all their embeddings |
| `delete_embedding()` | Removes a single reference photo from a person's profile |
| `merge_identities()` | Moves all embeddings from source identity into target identity, then deletes the source — used when the user tries to enroll a duplicate name |
| `log_detection()` | Inserts a timestamped detection log entry |
| `get_detection_logs()` | Queries logs with optional category filter and name search, ordered by newest first |
| `delete_detection_log()` / `clear_all_detection_logs()` | Log table management |

#### `install.py` (~55 lines)
Cross-platform installer that detects your OS (`platform.system()`) and runs the correct pip commands:
- **Windows**: Installs `PyQt6<6.6`, `opencv-python-headless`, PyTorch 2.2.2 with CUDA 11.8 index, `facenet-pytorch==2.5.3`, `numpy<2.0.0`, `pillow`
- **macOS**: Standard PyTorch (auto-enables MPS on Apple Silicon), same other deps
- **Linux**: Same as macOS path

#### `build.spec` (~67 lines)
PyInstaller specification file:
- Entry point: `main.py`
- Collects torch dynamic libraries via `collect_dynamic_libs('torch')`
- Explicit `hiddenimports` for `torch`, `torchvision`, `facenet_pytorch`, `numpy`, `cv2`, `db_manager`, `face_pipeline`
- Points to custom `hookspath=['hooks']` for model weight collection
- Excludes unused heavy packages: `matplotlib`, `scipy`, `IPython`, `notebook`
- Enables UPX compression if available
- Output name: `FaceRecognitionCCTV`

#### `hooks/hook-facenet_pytorch.py` (~8 lines)
PyInstaller hook that calls `collect_submodules('facenet_pytorch')` and `collect_data_files('facenet_pytorch')` to ensure the pretrained `.pt` model weight files are bundled into the compiled executable.

#### `fix_db.py` (~33 lines)
Standalone repair script.  If `face_db.sqlite` has broken foreign keys (caused by SQLite `ALTER TABLE RENAME` quirks during schema migrations), this script detects the corruption, rebuilds the `embeddings` table with correct `FOREIGN KEY` constraints, and migrates all data.  Run via `python fix_db.py` if the app crashes with FK errors.

---

## 🧠 Deep-Learning Pipeline

The inference pipeline runs inside `face_pipeline.py` and is called by every thread:

```
Raw Camera Frame (BGR, OpenCV)
        │
        ▼
  cv2.cvtColor → RGB
        │
        ▼
  Image.fromarray() → PIL Image
        │
        ▼
┌───────────────────────────────┐
│  MTCNN (Multi-task Cascaded   │
│  Convolutional Networks)      │
│  ─ 3-stage cascaded detector  │
│  ─ P-Net → R-Net → O-Net     │
│  ─ min_face_size = 40px       │
│  ─ thresholds = [0.6,0.7,0.7]│
│  ─ keep_all = True            │
│  ─ Runs on GPU (cuda/mps)     │
│  Output: bounding boxes +     │
│          confidence probs     │
└───────────────┬───────────────┘
                │
                ▼
       Crop faces from frame
                │
                ▼
┌───────────────────────────────┐
│  InceptionResnetV1            │
│  ─ Pretrained on VGGFace2    │
│  ─ Input: 160×160 face crop  │
│  ─ Output: 512-dim float32   │
│    embedding vector           │
│  ─ torch.no_grad() for speed │
└───────────────┬───────────────┘
                │
                ▼
   L2-Normalize embedding
                │
                ▼
┌───────────────────────────────┐
│  Cosine Similarity Matching   │
│  ─ dot(emb, db_emb) for each │
│    enrolled vector in SQLite  │
│  ─ Threshold: 0.75 (strict)  │
│  ─ Best match wins the label │
└───────────────────────────────┘
                │
                ▼
   Return: {box, name, category,
            similarity, embedding}
```

### Why Two MTCNN Instances?
- `self.mtcnn` (`keep_all=True`): Used during live/batch processing to detect **all** faces in a frame simultaneously
- `self.mtcnn_single` (`keep_all=False`): Used during manual enrollment to extract exactly **one** face from an uploaded photo, ensuring clean single-face embeddings

---

## 📐 Face Tracking Algorithm

Tracking ensures that when the system sees "Person A on the left and Person B on the right" in frame N, it correctly associates the same physical people in frame N+1 even if they moved.

The function `compute_tracking_score(boxA, boxB)` computes:

```python
score = IoU(boxA, boxB) + max(0, 1.0 - normalized_centroid_distance)
```

**Step 1 — Intersection over Union (IoU):**
```
IoU = Area_of_Overlap / Area_of_Union
```
Measures how much two bounding boxes physically overlap.  Identical boxes = 1.0, no overlap = 0.0.

**Step 2 — Normalized Centroid Distance:**
```
centroid_dist = sqrt((cx_A - cx_B)² + (cy_A - cy_B)²)
normalized = centroid_dist / max(width_A, width_B)
penalty = max(0, 1.0 - normalized)
```
Prevents fragmentation when a subject moves their head rapidly (box shifts but IoU drops).  The centroid check adds resilience.

**Matching Logic:**
- Each existing track tries to match the detected box with the highest combined score
- Score must exceed `0.4` to be considered a valid match
- Unmatched boxes become **new tracks**
- Tracks that go unmatched for **5 consecutive frames** are "finished" (the person left the view)

---

## 🗳️ Identity Locking (Temporal Voting)

A single frame can produce a noisy or incorrect classification (bad angle, motion blur, partial occlusion).  The system prevents this from poisoning the display using a **rolling-window majority vote**:

1. Each track maintains a `history` list capped at **30 entries** (roughly 1 second of footage at 30 FPS)
2. Every frame, the per-frame classification result is appended as a vote: either `(identity_id, name, category)` or `(None, 'Unknown', 'Unknown')`
3. Only votes where `similarity > 0.65` count as "known" votes (lower than the strict 0.75 batch threshold — the temporal system compensates for occasional weak frames)

**Initial Latch:** If a track is currently labeled "Unknown" and accumulates **≥ 3 known votes** for the same identity, the track **locks** onto that identity.

**Identity Swap:** If a track is already latched and a **different** identity accumulates **≥ 5 votes**, only then does the label switch — preventing brief glitches from overriding a correct identification.

This means: even if your face only matches the database in 3 out of 30 frames, the system will still identify you reliably.  And a single bad frame where you momentarily resemble someone else will **never** cause a misidentification.

---

## 🔄 Post-Processing: Matching & Clustering

When batch processing ends (video file finishes, or user clicks "Finish & Save" in Hybrid mode), the system runs a multi-stage aggregation pipeline:

### Stage 1: Track Filtering
Tracks with fewer than **3 frames** of face data are discarded as noise (brief detections, false positives).

### Stage 2: Known Identity Matching
For each surviving track:
1. Select ~5 representative photos at evenly spaced intervals across the track's lifetime (using `np.linspace`)
2. Compare **every embedding in the track** against **every embedding of every identity in the database**
3. Compute cosine similarity: `sim = dot(track_emb_normalized, db_emb_normalized)`
4. If the best similarity across all comparisons exceeds **0.75**:
   - The track is matched to that known identity
   - The ~5 representative photos are saved to `captured_faces/` and their embeddings are added to that identity's profile in the database
   - This **automatically enriches** your database with more angles of known people!

### Stage 3: Unknown Clustering
Tracks that don't match any known identity are compared **against each other**:
1. For each unmatched track, compute its best cosine similarity against every other unmatched track
2. If similarity exceeds **0.72**, the tracks are grouped together (they're the same anonymous person)
3. This prevents the system from creating 5 separate "Unknown" entries for the same stranger who walked past the camera multiple times

### Stage 4: Database Commit
Each cluster of unknown tracks is saved as a new identity named `Unknown_Group_XXXX` (or `Unknown_Hybrid_XXXX` for hybrid mode) with category "Unknown".  The representative photos are written to `captured_faces/` and linked in the database.

---

## ⏱️ Detection Log Cooldown

In **Detection Only** mode, when a VIP or Blacklist person is recognized, the system logs their presence to the `detection_logs` table.  However, if someone stands in front of the camera for 20 minutes, you don't want 36,000 log entries.

**How it works:**
- `FaceVideoThread` maintains an in-memory dictionary: `last_logged_identities = {identity_id: timestamp}`
- When a VIP/Blacklist face is identified, the system checks: `current_time - last_logged_time > 600` (600 seconds = 10 minutes)
- If the cooldown has expired (or this is the first sighting), the detection is logged and the timer resets
- If the cooldown is still active, the detection is silently skipped

This ensures exactly **one log entry per 10-minute window per person**, regardless of how long they remain on camera.

---

## 🖥️ User Interface

The application is organized into **3 primary tabs**, with the Database tab containing **3 sub-tabs**:

### Tab 1: Live Recognition
The main real-time monitoring console.

**Layout:**
- Large video preview label (fills most of the window, scales responsively)
- Two side-by-side buttons at the bottom:
  - **"Start Detection Only"** → Launches `FaceVideoThread`. Draws colored bounding boxes on each face with name labels. Colors: Green = VIP, Red = Blacklist, Gray = Unknown. Logs VIP/Blacklist entries with cooldown. No data is saved to the database beyond logs.
  - **"Start Detection + Capture"** → Launches `HybridCaptureThread`. Same visual experience as Detection Only, but silently captures face crops and embeddings in the background. When stopped via **"Finish & Save Data"**, runs the full matching + clustering pipeline and saves everything to the database.
- Hidden progress bar and log widget that appear during post-capture aggregation

**Mirror Flip:** The video feed is horizontally flipped (`cv2.flip(frame, 1)`) so it feels like looking in a mirror. Bounding box coordinates are mathematically mirrored (`x1 = width - box[2]`) to stay aligned.

### Tab 2: Batch Processing
A dedicated interface for processing pre-recorded videos or capturing from camera without the overhead of real-time box rendering.

**Layout:**
- Centered title: "Batch Video/Camera Face Processing"
- Description text explaining the feature
- Large **"Start Batch Processing"** button that opens a dialog asking:
  - **"Video File"** → Opens a file picker for `.mp4`, `.avi`, `.mkv`, `.mov` files
  - **"Live Camera"** → Opens camera index 0
  - **"Cancel"** → Dismisses

**Processing Dialog (`VideoProcessorDialog`):**
- For video files: shows filename and a percentage progress bar
- For camera: shows "Processing: Live Camera 0", a live video preview feed, an indeterminate progress bar, and a "Stop Capture" button
- Log widget shows real-time processing messages
- "Close" button activates after processing completes

### Tab 3: Database Management
Contains 3 sub-tabs:

#### Sub-tab: Registered Identities
- Grid of profile cards (220×260px each) for all VIP and Blacklist entries
- Each card shows: thumbnail of first enrolled photo, bold name, colored category label (green for VIP, red for Blacklist), photo count
- **"Enrol New Person"** button → Opens `AddEditIdentityDialog`
- **"Edit"** button on each card → Opens `AddEditIdentityDialog` pre-filled with existing data
- **"Delete"** button on each card → Confirmation dialog → Removes identity + all embeddings (CASCADE)
- **"Refresh List"** button → Re-queries database and rebuilds the grid

#### Sub-tab: Unknown Personalities
- Identical grid layout showing all "Unknown" category entries
- These are auto-generated by batch processing and hybrid capture
- Each card has **Edit** and **Delete** buttons
- Clicking **Edit** on an Unknown opens the identity dialog where you can:
  - Change the auto-generated name to a real name
  - Change category from "Unknown" to "VIP" or "Blacklist"
  - Upload additional reference photos
  - Delete bad reference photos
  - This effectively **promotes** an unknown stranger into a known, tracked identity

#### Sub-tab: Detection Logs
- Search bar: filter logs by name (partial match)
- Category dropdown: filter by "All", "VIP", or "Blacklist"
- Table with columns: Name, Category, Timestamp (sorted newest first)
- **"Delete Selected Log"** → Removes one entry
- **"Clear All Logs"** → Purges entire log table (with confirmation)

### The Enrollment Dialog (`AddEditIdentityDialog`)
This is the most feature-rich dialog in the app.  It appears when creating a new identity or editing an existing one.

**Fields:**
- **Name input** (text field, pre-filled when editing)
- **Category dropdown**: "VIP" / "Blacklist" (or "Unknown" / "VIP" / "Blacklist" if editing an Unknown)

**Photo Management:**
- **"Upload Files"** button → Opens multi-file picker for `.png`, `.jpg`, `.jpeg`. You can select dozens of images at once. Each appears as a thumbnail in the gallery below.
- **"Take Photo via Camera"** button → Opens `CaptureDialog`: a live webcam preview with a "Take Snapshot" button. The captured frame is saved to a temp file and added to the gallery.
- **Gallery grid** shows all photos (existing + pending). Each has a **"Remove"** button.
  - For existing photos: removing deletes the embedding from the database immediately
  - For pending photos: removing just drops them from the upload queue
- **"Save & Enroll"** button → Processes all pending images through `FaceProcessor.get_embedding()`, validates a face was detected in each, and saves to DB

**Duplicate Detection & Merge:**
When saving, if an identity with the same name already exists, the system prompts:
- "Do you want to merge this profile into the existing one?"
- If yes, all embeddings are transferred to the existing profile
- If there's a category conflict (e.g., existing is VIP, new is Blacklist), a second prompt asks which category to keep

---

## 🗄️ Database Architecture

The database is a single `face_db.sqlite` file using three tables:

### Schema

```sql
CREATE TABLE identities (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL,
    category TEXT NOT NULL CHECK(category IN ('VIP', 'Blacklist', 'Unknown'))
);

CREATE TABLE embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER NOT NULL,
    image_path  TEXT,
    embedding   array,           -- Custom type: numpy array serialized as BLOB
    FOREIGN KEY(identity_id) REFERENCES identities(id) ON DELETE CASCADE
);

CREATE TABLE detection_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER,
    name        TEXT,
    category    TEXT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Numpy ↔ SQLite Serialization
The `DatabaseManager` registers custom SQLite adapters:
- **`adapt_array(ndarray)`**: Serializes a numpy array → `BytesIO` → `sqlite3.Binary` blob using `np.save()`
- ****`convert_array(blob)`**: Deserializes binary blob → `BytesIO` → numpy array using `np.load()`

This lets you write `c.execute("INSERT ... VALUES (?, ?)", (id, numpy_embedding))` and SQLite transparently handles the conversion.

### Schema Migrations
The database self-migrates across three schema versions:
- **V1 → V2**: Original single-embedding-per-identity schema → multi-embedding (separate `embeddings` table)
- **V2 → V3**: Added 'Unknown' to the category CHECK constraint
- **FK Repair**: Auto-detects and fixes broken foreign keys caused by `ALTER TABLE RENAME` (a well-known SQLite limitation)

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.9 – 3.11**
- A webcam (built-in or USB)
- *Recommended*: NVIDIA GPU with CUDA drivers **or** Apple Silicon Mac (M1/M2/M3/M4) for GPU acceleration

### Step-by-Step

```bash
# 1. Clone the repository
git clone <repo-url>
cd face_recognition_app

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Run the automated installer (detects your OS & GPU)
python install.py

# 5. Launch the application
python main.py
```

**First launch notes:**
- The console will print `FaceProcessor running on device: cuda:0` (or `mps` or `cpu`)
- PyTorch will download ~100MB of pretrained VGGFace2 weights on first run (cached for future launches)
- `face_db.sqlite` is auto-created in the project directory
- `captured_faces/` directory is auto-created for storing cropped face images

### Manual Installation (if `install.py` fails)
```bash
pip install PyQt6<6.6 opencv-python-headless numpy<2.0.0 pillow facenet-pytorch==2.5.3
# For CUDA (Windows/Linux with NVIDIA GPU):
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
# For CPU only or macOS:
pip install torch torchvision torchaudio
```

### Database Repair
If the app crashes with foreign key errors after a schema migration:
```bash
python fix_db.py
```

---

## 📦 Building the Standalone Desktop App

You can compile the entire application — Python runtime, PyTorch models, OpenCV, PyQt6, and all dependencies — into a single distributable folder that runs on any Windows machine without Python installed.

### Prerequisites
```bash
pip install pyinstaller
```

### Build Steps

```bash
# 1. Run PyInstaller with the provided spec file
pyinstaller build.spec

# 2. Wait for compilation (may take 5-15 minutes depending on system)
# PyInstaller will:
#   - Collect all torch dynamic libraries (.dll/.so)
#   - Bundle facenet_pytorch model weights via hooks/hook-facenet_pytorch.py
#   - Exclude unnecessary packages (matplotlib, scipy, IPython)
#   - Apply UPX compression if available

# 3. Find your compiled application
cd dist/FaceRecognitionCCTV/

# 4. Run it!
FaceRecognitionCCTV.exe
```

### What's Inside `build.spec`
- **Entry point**: `main.py`
- **Dynamic libraries**: `collect_dynamic_libs('torch')` — essential for torch to find its CUDA/CPU backends at runtime
- **Hidden imports**: Explicitly lists `torch`, `torchvision`, `facenet_pytorch`, `numpy`, `cv2`, `db_manager`, `face_pipeline` so PyInstaller doesn't miss them during static analysis
- **Custom hooks**: `hookspath=['hooks']` points to `hook-facenet_pytorch.py` which calls `collect_submodules()` and `collect_data_files()` to bundle the pretrained `.pt` weight files
- **Exclusions**: `matplotlib`, `scipy`, `IPython`, `notebook` are excluded to reduce bundle size
- **Console mode**: `console=True` is set by default so you can see error messages if the app crashes. Change to `console=False` for a clean windowed release.

### Distributing
1. Zip the entire `dist/FaceRecognitionCCTV/` folder
2. Send the zip to any Windows machine
3. The recipient extracts it and double-clicks `FaceRecognitionCCTV.exe`
4. No Python, no pip, no setup — it just works

> **Tip:** To add a custom application icon, place a `.ico` file in the project root and set `icon='your_icon.ico'` in the `EXE()` block of `build.spec`.
