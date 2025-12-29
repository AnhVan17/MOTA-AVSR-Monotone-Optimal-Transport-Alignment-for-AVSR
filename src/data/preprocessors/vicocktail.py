import os
import glob
from .base import BasePreprocessor

class ViCocktailPreprocessor(BasePreprocessor):
    def collect_metadata(self):
        print(f"   [ViCocktail] Scanning for files in {self.data_root}...")
        
        # Scan for common video formats
        video_extensions = ['*.mp4', '*.webm', '*.mkv', '*.avi']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.data_root, "**", ext), recursive=True))
            
        results = []
        for video_path in video_files:
            # Assume transcript has same basename but .txt extension
            text = self._get_transcript(video_path)
            rel_path = os.path.relpath(video_path, self.data_root)
            filename = os.path.splitext(os.path.basename(video_path))[0]
            
            results.append({
                'id': filename,
                'full_path': video_path,
                'rel_path': rel_path,
                'text': text
            })
            
        return results

    def _get_transcript(self, video_path):
        """
        Look for a corresponding text file.
        Strategy: 
        1.  Same name .txt in same folder.
        2.  Same name .txt in 'labels' or 'transcripts' subfolder (optional, can add if needed).
        """
        base_path = os.path.splitext(video_path)[0]
        txt_path = base_path + ".txt"
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except:
                pass
                
        return ""
