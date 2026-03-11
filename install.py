import os
import sys
import subprocess
import platform

def run_command(command):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    print("Starting Setup for Face Recognition CCTV App...")
    
    # Core dependencies that are cross-platform
    core_deps = [
        "numpy<2.0.0",
        "facenet-pytorch==2.5.3",
        "pillow"
    ]
    
    os_name = platform.system().lower()
    
    if os_name == "windows":
        print("Detected OS: Windows")
        # Windows-specific dependencies (dealing with PyQt6 DLL issues and PyTorch CUDA compatibility)
        install_cmds = [
            f"{sys.executable} -m pip install PyQt6<6.6 opencv-python-headless",
            f"{sys.executable} -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118",
            f"{sys.executable} -m pip install " + " ".join(core_deps)
        ]
        
    elif os_name == "darwin":  # macOS
        print("Detected OS: macOS (Apple Silicon or Intel)")
        # macOS handles PyQt6 and torch flawlessly usually
        install_cmds = [
            f"{sys.executable} -m pip install PyQt6 opencv-python-headless",
            f"{sys.executable} -m pip install torch torchvision torchaudio", # Standard torch gets MPS support on modern macs
            f"{sys.executable} -m pip install " + " ".join(core_deps)
        ]
        
    else:  # Linux
        print("Detected OS: Linux")
        install_cmds = [
            f"{sys.executable} -m pip install PyQt6 opencv-python-headless",
            f"{sys.executable} -m pip install torch torchvision torchaudio",
            f"{sys.executable} -m pip install " + " ".join(core_deps)
        ]

    for cmd in install_cmds:
        run_command(cmd)
        
    print("\nSetup Complete! You can now run the application with: python main.py")

if __name__ == "__main__":
    main()
