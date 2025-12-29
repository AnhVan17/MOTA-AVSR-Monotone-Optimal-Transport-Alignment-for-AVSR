import modal
import os

APP_NAME = "check-volume-structure"
VOLUME_NAME = "avsr-dataset-volume"
VOL_MOUNT_PATH = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(volumes={VOL_MOUNT_PATH: volume})
def inspect_volume():
    print(f"📂 Checking contents of {VOL_MOUNT_PATH}...")

    # Hàm để in cây thư mục đệ quy
    def list_files(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            # Chỉ in 5 file đầu tiên mỗi thư mục để đỡ rối
            for f in files[:5]:
                print(f'{subindent}{f}')
            if len(files) > 5:
                print(f'{subindent}... ({len(files)-5} more files)')

    list_files(VOL_MOUNT_PATH)


@app.local_entrypoint()
def main():
    inspect_volume.remote()
