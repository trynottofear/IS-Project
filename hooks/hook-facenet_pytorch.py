from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all modules from facenet_pytorch
hiddenimports = collect_submodules('facenet_pytorch')

# Collect the model weights and data files
datas = collect_data_files('facenet_pytorch')
