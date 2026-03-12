import cv2
import numpy as np
import mediapipe as mp
import os
# Standalone logger setup to avoid importing 'src.utils' which triggers 'torch' import
def setup_logger(name):
    import logging
    import sys
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger(__name__)

class FaceMeshConfig:
    IMAGE_SIZE = 88
    FRAME_RATE = 25
    # Standardize output format
    OUTPUT_EXT = ".mp4" 

class FaceMeshPreprocessor:
    """
    Dedicated FaceMesh Processor for CPU-based encoding.
    Designed to be run as a Singleton or with minimal re-initialization.
    """
    _instance = None
    _face_mesh = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaceMeshPreprocessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure FaceMesh is initialized only once
        if FaceMeshPreprocessor._face_mesh is None:
            self._init_facemesh()

    def _init_facemesh(self):
        """
        Initialize MediaPipe FaceMesh on CPU.
        """
        logger.info("Initializing MediaPipe FaceMesh (CPU Mode)...")
        
        # Force CPU to avoid EGL/GPU conflicts
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            FaceMeshPreprocessor._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3, # Lower threshold for wild videos
                min_tracking_confidence=0.3
            )
        finally:
             if original_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
             else:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        
        logger.info("FaceMesh Initialized Successfully.")

    def process_video(self, video_path, output_path=None):
        """
        Process a single video: Detect Mouth -> Crop -> Return Frames or Save.
        """
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or FaceMeshConfig.FRAME_RATE
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if not frames:
            return None

        # Process Frames
        cropped_frames = []
        prev_bbox = None
        
        for i, frame in enumerate(frames):
            # Optimization: Detect every 5 frames to speed up CPU processing
            if i % 5 == 0 or prev_bbox is None:
                crop, prev_bbox = self._extract_mouth(frame, None)
            else:
                crop, prev_bbox = self._extract_mouth(frame, prev_bbox)
            
            # Resize
            crop = cv2.resize(crop, (FaceMeshConfig.IMAGE_SIZE, FaceMeshConfig.IMAGE_SIZE))
            cropped_frames.append(crop)

        # Convert to numpy array [T, H, W, C]
        cropped_frames = np.array(cropped_frames)

        # Output Handling
        if output_path:
            self._save_video(cropped_frames, output_path, fps)
            return output_path
        else:
            return cropped_frames

    def _extract_mouth(self, frame, prev_bbox):
        """Core landmarks logic."""
        h, w = frame.shape[:2]
        
        if prev_bbox:
            x1, y1, x2, y2 = prev_bbox
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                return frame[y1:y2, x1:x2], prev_bbox

        try:
            # MediaPipe needs RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = FaceMeshPreprocessor._face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # Mouth Indices (mp.solutions.face_mesh.FACEMESH_LIPS is complex, using manual)
                mouth_idx = [13, 14, 61, 291, 78, 308, 324, 17] 
                
                xs = [lm[i].x * w for i in mouth_idx]
                ys = [lm[i].y * h for i in mouth_idx]
                
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                
                # Dynamic Radius
                radius = max(
                    int((max(xs) - min(xs)) * 1.8) // 2,
                    int((max(ys) - min(ys)) * 1.8) // 2,
                    FaceMeshConfig.IMAGE_SIZE // 2
                )
                
                y1 = max(0, cy - radius)
                y2 = min(h, cy + radius)
                x1 = max(0, cx - radius)
                x2 = min(w, cx + radius)
                
                return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        except Exception:
            pass
            
        # Fallback Center Crop
        cy, cx = h // 2, w // 2
        r = FaceMeshConfig.IMAGE_SIZE // 2
        return frame[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)], None

    def _save_video(self, frames, path, fps):
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
