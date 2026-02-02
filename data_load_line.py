import os
from huggingface_hub import snapshot_download


folder_path = snapshot_download(
    repo_id="Teklia/IAM-line",
    repo_type="dataset",
    local_dir="./data/IAM-line",
    local_dir_use_symlinks=False
)

print(f"Download thành công! Các file nằm tại: {os.path.abspath(folder_path)}")


for root, dirs, files in os.walk(folder_path):
    for file in files:
        print(os.path.join(root, file))