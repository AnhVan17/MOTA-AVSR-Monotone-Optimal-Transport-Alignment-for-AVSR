import cv2
import numpy as np
import mediapipe as mp
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Standalone configuration for cropping to avoid circular imports or heavy dependencies
class CropperConfig:
    IMAGE_SIZE = 88          # Mouth crop size (SOTA)
    RESNET_INPUT_SIZE = 88   # Changed from 224 to 88 (SOTA standard)
    FRAME_RATE = 25          # GRID dataset is 25fps

class MouthCropper:
    """
    Reads a video, detects mouth, crops, and saves as a new video file.
    """
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def process_video(self, input_path, output_path):
        """
        Reads input_path (.mpg), crops mouth, writes to output_path (.mp4)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open {input_path}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or CropperConfig.FRAME_RATE
        
        # Prepare Video Writer
        # avc1 is H.264, good for mp4. If fails on linux headless without openh264, try mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # We output at 224x224 so it is ready for ResNet
        out = cv2.VideoWriter(output_path, fourcc, fps, (CropperConfig.RESNET_INPUT_SIZE, CropperConfig.RESNET_INPUT_SIZE))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if not frames:
            return False

        prev_bbox = None
        
        for i, frame in enumerate(frames):
            # Optimization: Detect every 5 frames
            if i % 5 == 0 or prev_bbox is None:
                crop, prev_bbox = self.extract_mouth(frame, None)
            else:
                crop, prev_bbox = self.extract_mouth(frame, prev_bbox)
            
            # Resize to 224x224 for ResNet18
            crop = cv2.resize(crop, (CropperConfig.RESNET_INPUT_SIZE, CropperConfig.RESNET_INPUT_SIZE)) 
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Note: VideoWriter expects BGR, but we converted to RGB for MediaPipe/Standard?
            # OpenCV VideoWriter expects BGR.
            # MediaPipe needed RGB.
            # extract_mouth returns "frame" slice which is... BGR (original).
            # But wait, inside extract_mouth:
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # return frame[ y1:y2 , ... ] -> This returns BGR slice of original frame.
            
            # So `crop` is BGR.
            # `cv2.resize` returns BGR.
            # `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` makes it RGB.
            # `out.write` expects BGR.
            # So we should NOT convert to RGB before writing to video if we want standard video.
            # However, if downstream ResNet expects RGB...
            # Standard video players expect BGR/YUV.
            # Let's save as BGR (standard video). DataLoader can convert to RGB.
            
            # REVERTING RGB CONVERSION FOR SAVING VIDEO
            # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) 
            
            out.write(crop)

        out.release()
        return True

        out.release()
        return True

    def extract_mouth(self, frame, prev_bbox=None):
        """Mouth cropping logic matching BasePreprocessor"""
        h, w = frame.shape[:2]
        
        # If previous bbox exists, try fast crop
        if prev_bbox is not None:
            x1, y1, x2, y2 = prev_bbox
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                return frame[y1:y2, x1:x2], prev_bbox

        # If not available or need to re-detect -> Run MediaPipe
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # Landmark points around the mouth
                mouth_idx = [13, 14, 61, 291, 78, 308, 324, 17]
                xs = [lm[i].x * w for i in mouth_idx]
                ys = [lm[i].y * h for i in mouth_idx]
                
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                
                # Calculate crop radius
                radius = max(
                    int((max(xs) - min(xs)) * 1.8) // 2,
                    int((max(ys) - min(ys)) * 1.8) // 2,
                    CropperConfig.IMAGE_SIZE // 2
                )
                
                y1 = max(0, cy - radius)
                y2 = min(h, cy + radius)
                x1 = max(0, cx - radius)
                x2 = min(w, cx + radius)
                
                bbox = (x1, y1, x2, y2)
                return frame[y1:y2, x1:x2], bbox
        except:
            pass
            
        # Fallback: Crop center
        cy, cx = h // 2, w // 2
        r = CropperConfig.IMAGE_SIZE // 2
        return frame[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)], None
