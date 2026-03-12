import glob
import os
import argparse
import json
from tqdm import tqdm

def merge_manifests(data_root: str, output_path: str, subset_keyword: str):
    """
    Merge multiple shard manifests into ONE.
    """
    print(f"Searching for manifests in {data_root} with keyword '{subset_keyword}'...")
    
    # 1. Find all files ending in .jsonl recursively
    all_files = glob.glob(os.path.join(data_root, "**/*.jsonl"), recursive=True)
    
    # 2. Filter by keyword (e.g. 'train' or 'test') AND ensure it's a shard manifest (usually ends with _manifest.jsonl)
    manifests = [
        f for f in all_files 
        if subset_keyword in os.path.basename(f) 
        and "merged" not in f # Avoid re-merging previous output
    ]
    
    if not manifests:
        print(f"No manifests found for '{subset_keyword}'!")
        return

    print(f"Found {len(manifests)} manifests. Merging...")
    
    # 3. Merge
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for m_path in tqdm(manifests):
            try:
                with open(m_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                            total_lines += 1
            except Exception as e:
                print(f"Error reading {m_path}: {e}")
                
    print(f"✅ Successfully merged {len(manifests)} files into {output_path}")
    print(f"   Total samples: {total_lines}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/vicocktail_features", help="Root search dir")
    parser.add_argument("--output_path", type=str, required=True, help="Output merged file path")
    parser.add_argument("--keyword", type=str, required=True, help="Keyword to filter manifests (e.g. 'train')")
    
    args = parser.parse_args()
    merge_manifests(args.data_root, args.output_path, args.keyword)
