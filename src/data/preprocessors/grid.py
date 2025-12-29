import os
import glob
from .base import BasePreprocessor

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class GridPreprocessor(BasePreprocessor):
    def collect_metadata(self):
        # GRID: Recursively find all video files (.mpg or .mp4)
        print(f"   [Grid] Scanning for video files in {self.data_root}...")
        
        # Support both original .mpg and cropped .mp4
        extensions = ['*.mpg', '*.mp4']
        video_files = []
        for ext in extensions:
             video_files.extend(glob.glob(os.path.join(self.data_root, "**", ext), recursive=True))
             
        print(f"   [Grid] Found {len(video_files)} video files.")
        
        # Parallel Processing Function
        def process_one(video_path):
            text = self._get_grid_transcript(video_path)
            rel_path = os.path.relpath(video_path, self.data_root)
            filename = os.path.splitext(os.path.basename(video_path))[0]
            return {
                'id': filename,
                'full_path': video_path,
                'rel_path': rel_path,
                'text': text
            }

        # Run with ThreadPool
        print(f"   [Grid] Metadata check running with multiple threads...")
        results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Use list() to trigger execution and tqdm to show progress
            results = list(tqdm(
                executor.map(process_one, video_files), 
                total=len(video_files), 
                desc="   [Grid] Metadata",
                unit="files"
            ))
            
        return results

    def _get_grid_transcript(self, video_path):
        """
        Logic to find GRID-specific align file.
        Handles both raw /data/grid (mpg) and /data/grid_cropped (mp4) cases.
        """
        try:
            parent_dir = os.path.dirname(video_path)
            filename = os.path.basename(video_path)
            base_name = os.path.splitext(filename)[0] # remove .mpg or .mp4
            
            # Construct candidate align paths
            candidates = []
            
            # 1. Standard /data/grid/s1/align/ -> align is sibling folder
            # video: /data/grid/s1/video.mpg
            # align: /data/grid/s1/align/video.align
            candidates.append(os.path.join(parent_dir, "align", base_name + ".align"))
            
            # 2. In-place (old structure)
            candidates.append(os.path.join(parent_dir, base_name + ".align"))
            
            # 3. Cross-reference from grid_cropped -> grid
            # video: /data/grid_cropped/s1/video.mp4
            # align: /data/grid/s1/align/video.align
            if "grid_cropped" in parent_dir:
                 # Replace grid_cropped with grid to find original source
                 original_parent = parent_dir.replace("grid_cropped", "grid")
                 candidates.append(os.path.join(original_parent, "align", base_name + ".align"))

            align_path = None
            for c in candidates:
                if os.path.exists(c):
                    align_path = c
                    break
            
            if align_path:
                words = []
                with open(align_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        # GRID format: start end word
                        if len(parts) >= 3:
                            word = parts[2]
                            if word not in ['sil', 'sp']:
                                words.append(word)
                return " ".join(words)
        except Exception:
            pass
        return ""