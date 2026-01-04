import modal
from pathlib import Path

app = modal.App("check-manifests")
volume = modal.Volume.from_name("avsr-dataset-volume")

@app.function(volumes={"/data": volume})
def check_manifest(query_text: str):
    import json
    
    test_manifest = "/data/manifests/test.jsonl"
    print(f"Checking {test_manifest} for '{query_text}'...")
    
    if not Path(test_manifest).exists():
        print("Test manifest not found.")
        return

    found = []
    with open(test_manifest, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            if query_text.lower() in item['text'].lower():
                found.append(item)
                print(f"\nMatch {len(found)} at line {i+1}:")
                print(f"  ID: {item['id']}")
                print(f"  Path: {item['path']}")
                print(f"  Text: {item['text']}")
            
            if len(found) >= 5: break
            
    if not found:
        print("No matches found.")

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "mười bài học"
    with app.run():
        check_manifest.remote(query)
