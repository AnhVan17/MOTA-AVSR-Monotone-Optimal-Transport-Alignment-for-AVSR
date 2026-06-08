import modal
import os

image = modal.Image.debian_slim()
app = modal.App("find-manifest")
volume = modal.Volume.from_name("avsr-volume")

@app.function(image=image, volumes={"/mnt": volume})
def list_files():
    print("Searching for .jsonl manifests in /mnt:")
    for root, dirs, files in os.walk("/mnt"):
        for name in files:
            if name.endswith(".jsonl"):
                print(os.path.join(root, name))

@app.local_entrypoint()
def main():
    list_files.remote()
