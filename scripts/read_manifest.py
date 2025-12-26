import modal
from pathlib import Path

app = modal.App("read-manifest")
volume = modal.Volume.from_name("avsr-dataset-volume")

@app.function(volumes={"/data": volume})
def read_head():
    import json
    test_manifest = "/data/manifests/test.jsonl"
    if not Path(test_manifest).exists():
        print("Test manifest not found.")
        return

    with open(test_manifest, 'r', encoding='utf-8') as f:
        f.readline() # Skip 0
        line1 = f.readline().strip()
        print("DEBUG_START")
        print(line1)
        print("DEBUG_END")

if __name__ == "__main__":
    with app.run():
        read_head.remote()
