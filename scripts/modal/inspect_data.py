import modal
import os
import subprocess
import glob

APP_NAME = "avsr-inspect-data"
VOLUME_NAME = "avsr-dataset-volume"
MOUNT_PATH = "/mnt"

image = modal.Image.debian_slim(python_version="3.10")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    timeout=600
)
def inspect_tar_content():
    input_dir = f"{MOUNT_PATH}/raw_mirror"
    
    print(f"🔍 Kiểm tra thư mục: {input_dir}")
    if not os.path.exists(input_dir):
        print(" Thư mục không tồn tại!")
        return

    # 1. Kiểm tra danh sách file và dung lượng
    tar_files = glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True)
    tar_files = sorted(tar_files)
    
    if not tar_files:
        print("Không tìm thấy file .tar nào!")
        return
        
    print(f"Tìm thấy {len(tar_files)} file tar.")
    
    # In thông tin 5 file đầu tiên
    print("\n--- 5 File đầu tiên ---")
    for tar in tar_files[:5]:
        size_mb = os.path.getsize(tar) / (1024 * 1024)
        print(f"📄 {os.path.basename(tar)}: {size_mb:.2f} MB")
        
        if size_mb < 0.1:
            print("   CẢNH BÁO: File quá nhỏ, có thể bị lỗi download!")

    # 2. Soi nội dung bên trong file đầu tiên
    sample_tar = tar_files[0]
    print(f"\n--- Nội dung bên trong {os.path.basename(sample_tar)} ---")
    try:
        # Chạy lệnh tar -tvf để liệt kê
        result = subprocess.run(["tar", "-tvf", sample_tar], capture_output=True, text=True)
        print(result.stdout[:2000]) # Chỉ in 2000 ký tự đầu
        
        if result.returncode != 0:
            print("Lỗi khi đọc file tar:")
            print(result.stderr)
    except Exception as e:
        print(f"Exception: {e}")

@app.local_entrypoint()
def main():
    inspect_tar_content.remote()
