"""
Production-ready Animal Detection Module
Uses YOLOv8m for efficient animal detection in images and videos
"""

from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for detection results"""
    has_animal: bool
    confidence: float
    detected_classes: List[str]
    bbox_count: int
    processing_time: float
    
    def to_dict(self) -> dict:
        return {
            "has_animal": self.has_animal,
            "confidence": self.confidence,
            "detected_classes": self.detected_classes,
            "bbox_count": self.bbox_count,
            "processing_time": self.processing_time
        }

class AnimalDetector:
    """
    Efficient animal detector for production use.
    Optimized for YOLOv8m based on performance testing.
    """
    
    # COCO dataset animal classes
    ANIMAL_CLASSES = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 
        18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 
        22: "zebra", 23: "giraffe"
    }
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the animal detector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        try:
            self.model = YOLO(model_path)
            if device:
                self.model.to(device)
            logger.info(f"✓ Model loaded: {model_path} on {self.model.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _process_detections(self, results) -> Tuple[bool, List[str], float, int]:
        """
        Process YOLO results and extract animal detections.
        
        Returns:
            Tuple of (has_animal, detected_classes, max_confidence, bbox_count)
        """
        has_animal = False
        detected_classes = []
        max_confidence = 0.0
        bbox_count = 0
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                
                if cls_id in self.ANIMAL_CLASSES:
                    has_animal = True
                    class_name = self.ANIMAL_CLASSES[cls_id]
                    if class_name not in detected_classes:
                        detected_classes.append(class_name)
                    max_confidence = max(max_confidence, conf)
                    bbox_count += 1
        
        return has_animal, detected_classes, max_confidence, bbox_count
    
    def detect_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        return_annotated: bool = False
    ) -> Union[DetectionResult, Tuple[DetectionResult, np.ndarray]]:
        """
        Detect animals in a single image.
        
        Args:
            image_path: Path to image file or numpy array (BGR format)
            return_annotated: If True, return annotated image along with results
            
        Returns:
            DetectionResult object, or tuple of (DetectionResult, annotated_image)
        """
        import time
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Process detections
            has_animal, classes, confidence, bbox_count = self._process_detections(results)
            
            processing_time = time.time() - start_time
            
            result = DetectionResult(
                has_animal=has_animal,
                confidence=confidence,
                detected_classes=classes,
                bbox_count=bbox_count,
                processing_time=processing_time
            )
            
            if return_annotated and results:
                annotated_image = results[0].plot()
                return result, annotated_image
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return DetectionResult(
                has_animal=False,
                confidence=0.0,
                detected_classes=[],
                bbox_count=0,
                processing_time=time.time() - start_time
            )
    
    # def detect_video(
    #     self,
    #     video_path: Union[str, Path],
    #     frame_skip: int = 2,
    #     max_frames: Optional[int] = None,
    #     save_output: Optional[str] = None,
    #     show_progress: bool = True
    # ) -> Dict:
    #     """
    #     Detect animals in a video with frame skipping for efficiency.
        
    #     Args:
    #         video_path: Path to video file
    #         frame_skip: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
    #         max_frames: Maximum number of frames to process (None = all frames)
    #         save_output: Path to save annotated video (None = don't save)
    #         show_progress: Print progress updates
            
    #     Returns:
    #         Dictionary with detection statistics and frame-by-frame results
    #     """
    #     import time
        
    #     cap = cv2.VideoCapture(str(video_path))
    #     if not cap.isOpened():
    #         logger.error(f"Failed to open video: {video_path}")
    #         return {"error": "Failed to open video"}
        
    #     # Video properties
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    #     # Limit frames if specified
    #     frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        
    #     # Video writer for output
    #     writer = None
    #     if save_output:
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
    #     # Statistics
    #     stats = {
    #             "total_frames": total_frames,
    #             "processed_frames": 0,
    #             "frames_with_animals": 0,
    #             "total_detections": 0,
    #             "confidence_sum": 0.0,          # NEW
    #             "confidence_count": 0,          # NEW
    #             "processing_time": 0,
    #             "detected_classes": set(),
    #             "frame_results": []
    #         }

        
    #     start_time = time.time()
    #     frame_idx = 0
    #     processed_count = 0
        
    #     try:
    #         while cap.isOpened() and (max_frames is None or processed_count < max_frames):
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
                
    #             # Process only every Nth frame
    #             if frame_idx % frame_skip == 0:
    #                 # Run detection
    #                 results = self.model(
    #                     source=frame,
    #                     conf=self.conf_threshold,
    #                     iou=self.iou_threshold,
    #                     verbose=False
    #                 )
                    
    #                 has_animal, classes, confidence, bbox_count = self._process_detections(results)
                    
    #                 stats["processed_frames"] += 1
    #                 processed_count += 1
                    
    #                 if has_animal:
    #                     stats["frames_with_animals"] += 1
    #                     stats["total_detections"] += bbox_count
    #                     stats["detected_classes"].update(classes)
    #                     stats["confidence_sum"] += confidence
    #                     stats["confidence_count"] += 1

                        
                        
    #                     # Store frame result
    #                     stats["frame_results"].append({
    #                         "frame_number": frame_idx,
    #                         "timestamp": frame_idx / fps,
    #                         "has_animal": True,
    #                         "confidence": confidence,
    #                         "classes": classes,
    #                         "bbox_count": bbox_count
    #                     })
                    
    #                 # Save annotated frame if requested
    #                 if writer and results:
    #                     annotated_frame = results[0].plot()
    #                     writer.write(annotated_frame)
                    
    #                 # Progress update
    #                 if show_progress and processed_count % 30 == 0:
    #                     progress = (processed_count / frames_to_process) * 100
    #                     logger.info(f"Progress: {progress:.1f}% | Animals detected in {stats['frames_with_animals']} frames")
                
    #             frame_idx += 1
            
    #         stats["processing_time"] = time.time() - start_time
    #         stats["detected_classes"] = list(stats["detected_classes"])
    #         stats["fps_processed"] = stats["processed_frames"] / stats["processing_time"] if stats["processing_time"] > 0 else 0
    #         stats["detection_rate"] = (stats["frames_with_animals"] / stats["processed_frames"] * 100) if stats["processed_frames"] > 0 else 0
    #         stats["avg_confidence"] = (
    #             stats["confidence_sum"] / stats["confidence_count"]
    #             if stats["confidence_count"] > 0 else 0.0
    #         )


            
    #     finally:
    #         cap.release()
    #         if writer:
    #             writer.release()
        
    #     if show_progress:
    #         logger.info(f"✓ Processed {stats['processed_frames']} frames in {stats['processing_time']:.2f}s")
    #         logger.info(f"✓ Detection rate: {stats['detection_rate']:.1f}%")
    #         logger.info(f"✓ Processing speed: {stats['fps_processed']:.1f} FPS")
        
    #     return stats


    def detect_video(
    self,
    video_path: Union[str, Path],
    frame_skip: int = 15,
    max_frames: Optional[int] = None,
    save_output: Optional[str] = None,
    show_progress: bool = True
    ) -> Dict:
        """
        Detect animals in a video with frame skipping for efficiency.
        """

        import time

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {"error": "Failed to open video"}

        # =====================================================
        # VIDEO PROPERTIES
        # =====================================================

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # =====================================================
        # LIMIT FRAMES
        # =====================================================

        frames_to_process = (
            min(total_frames, max_frames)
            if max_frames
            else total_frames
        )

        # =====================================================
        # VIDEO WRITER
        # =====================================================

        writer = None

        if save_output:

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            writer = cv2.VideoWriter(
                save_output,
                fourcc,
                fps,
                (width, height)
            )

        # =====================================================
        # STATS
        # =====================================================

        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "frames_with_animals": 0,
            "total_detections": 0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
            "processing_time": 0,
            "detected_classes": set(),
            "frame_results": []
        }

        start_time = time.time()

        frame_idx = 0

        processed_count = 0

        detection_start_time = None

        detection_end_time = None

        try:

            while (
                cap.isOpened()
                and (
                    max_frames is None
                    or processed_count < max_frames
                )
            ):

                ret, frame = cap.read()

                if not ret:
                    break

                # =============================================
                # PROCESS EVERY NTH FRAME
                # =============================================

                if frame_idx % frame_skip == 0:

                    # =========================================
                    # CURRENT VIDEO TIMESTAMP
                    # =========================================

                    timestamp_sec = (
                        cap.get(cv2.CAP_PROP_POS_MSEC)
                        / 1000
                    )

                    # =========================================
                    # RUN DETECTION
                    # =========================================

                    results = self.model(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )

                    (
                        has_animal,
                        classes,
                        confidence,
                        bbox_count
                    ) = self._process_detections(results)

                    stats["processed_frames"] += 1

                    processed_count += 1

                    # =========================================
                    # ANIMAL FOUND
                    # =========================================

                    if has_animal:

                        # =====================================
                        # DURATION TRACKING
                        # =====================================

                        if detection_start_time is None:
                            detection_start_time = timestamp_sec

                        detection_end_time = timestamp_sec

                        stats["frames_with_animals"] += 1

                        stats["total_detections"] += bbox_count

                        stats["detected_classes"].update(
                            classes
                        )

                        stats["confidence_sum"] += confidence

                        stats["confidence_count"] += 1

                        # =====================================
                        # STORE FRAME RESULT
                        # =====================================

                        stats["frame_results"].append({
                            "frame_number": frame_idx,
                            "timestamp": frame_idx / fps,
                            "has_animal": True,
                            "confidence": confidence,
                            "classes": classes,
                            "bbox_count": bbox_count
                        })

                    # =========================================
                    # SAVE ANNOTATED VIDEO
                    # =========================================

                    if writer and results:

                        annotated_frame = results[0].plot()

                        writer.write(annotated_frame)

                    # =========================================
                    # PROGRESS LOGGING
                    # =========================================

                    if (
                        show_progress
                        and processed_count % 30 == 0
                    ):

                        progress = (
                            processed_count
                            / frames_to_process
                        ) * 100

                        logger.info(
                            f"Progress: {progress:.1f}% | "
                            f"Animals detected in "
                            f"{stats['frames_with_animals']} "
                            f"frames"
                        )

                frame_idx += 1

            # =================================================
            # FINAL STATS
            # =================================================

            stats["processing_time"] = (
                time.time() - start_time
            )

            stats["detected_classes"] = list(
                stats["detected_classes"]
            )

            stats["fps_processed"] = (
                stats["processed_frames"]
                / stats["processing_time"]
                if stats["processing_time"] > 0
                else 0
            )

            stats["detection_rate"] = (
                (
                    stats["frames_with_animals"]
                    / stats["processed_frames"]
                ) * 100
                if stats["processed_frames"] > 0
                else 0
            )

            stats["avg_confidence"] = (
                stats["confidence_sum"]
                / stats["confidence_count"]
                if stats["confidence_count"] > 0
                else 0.0
            )

            # =============================================
            # DURATION PRINT
            # =============================================

            if (
                detection_start_time is not None
                and detection_end_time is not None
            ):

                duration = (
                    detection_end_time
                    - detection_start_time
                )

                logger.info(
                    f"Animal detected from "
                    f"{detection_start_time:.2f}s "
                    f"to {detection_end_time:.2f}s "
                    f"(duration: {duration:.2f}s)"
                )

        finally:

            cap.release()

            if writer:
                writer.release()

        # =====================================================
        # FINAL LOGS
        # =====================================================

        if show_progress:

            logger.info(
                f"✓ Processed "
                f"{stats['processed_frames']} "
                f"frames in "
                f"{stats['processing_time']:.2f}s"
            )

            logger.info(
                f"✓ Detection rate: "
                f"{stats['detection_rate']:.1f}%"
            )

            logger.info(
                f"✓ Processing speed: "
                f"{stats['fps_processed']:.1f} FPS"
            )

        return {
            "detected": (
                stats["frames_with_animals"] > 0
            ),

            "start_time": (
                round(detection_start_time, 2)
                if detection_start_time is not None
                else None
            ),

            "end_time": (
                round(detection_end_time, 2)
                if detection_end_time is not None
                else None
            ),

            "duration": (
                round(
                    detection_end_time
                    - detection_start_time,
                    2
                )
                if (
                    detection_start_time is not None
                    and detection_end_time is not None
                )
                else 0
            ),

            "frames_with_animals": (
                stats["frames_with_animals"]
            ),

            "processed_frames": (
                stats["processed_frames"]
            ),

            "avg_confidence": (
                round(
                    stats["avg_confidence"],
                    2
                )
            ),

            "detected_classes": (
                stats["detected_classes"]
            )
        }
    
    def is_animal(
    self,
    source: Union[str, Path, np.ndarray],
    quick_mode: bool = True
):
        """
        Robust unified function to check if source contains animals.

        IMAGE:
            Returns only True / False

        VIDEO:
            Returns:
            {
                detected,
                start_time,
                end_time,
                duration,
                detected_classes
            }
        """

        # ─────────────────────────────
        # IMAGE (numpy array)
        # ─────────────────────────────

        if isinstance(source, np.ndarray):

            result = self.detect_image(source)

            return result.has_animal

        source_path = Path(source)

        if not source_path.exists():

            logger.error(
                f"File not found: {source}"
            )

            return False

        video_extensions = {
            '.mp4',
            '.avi',
            '.mov',
            '.mkv',
            '.flv',
            '.wmv',
            '.webm'
        }

        image_extensions = {
            '.jpg',
            '.jpeg',
            '.png',
            '.bmp',
            '.tiff',
            '.webp'
        }

        file_ext = source_path.suffix.lower()

        # ─────────────────────────────
        # IMAGE
        # ─────────────────────────────

        if file_ext in image_extensions:

            result = self.detect_image(source)

            return result.has_animal

        # ─────────────────────────────
        # VIDEO
        # ─────────────────────────────

        if file_ext in video_extensions:

            frame_skip = 20 if quick_mode else 2

            max_frames = 50 if quick_mode else None

            video_stats = self.detect_video(
                source,
                frame_skip=frame_skip,
                max_frames=max_frames,
                show_progress=False
            )

            frames_with_animals = video_stats.get(
                "frames_with_animals",
                0
            )

            processed_frames = video_stats.get(
                "processed_frames",
                0
            )

            avg_confidence = video_stats.get(
                "avg_confidence",
                0.0
            )

            # ─────────────────────────
            # SAFETY CHECK
            # ─────────────────────────

            if processed_frames == 0:

                return {
                    "detected": False
                }

            # ─────────────────────────
            # SHORT VIDEO LOGIC
            # ─────────────────────────

            if processed_frames < 10:

                is_detected = (
                    frames_with_animals >= 2
                    and avg_confidence >= 0.4
                )

            # ─────────────────────────
            # NORMAL VIDEO LOGIC
            # ─────────────────────────

            else:

                min_required_frames = max(
                    2,
                    int(0.1 * processed_frames)
                )

                detection_rate = (
                    frames_with_animals
                    / processed_frames
                )

                is_detected = (
                    frames_with_animals
                    >= min_required_frames
                    and detection_rate >= 0.10
                    and avg_confidence >= 0.4
                )

            # ─────────────────────────
            # FINAL VIDEO RESPONSE
            # ─────────────────────────

            if is_detected:

                return {
                    "detected": True,

                    "start_time": video_stats.get(
                        "start_time"
                    ),

                    "end_time": video_stats.get(
                        "end_time"
                    ),

                    "duration": video_stats.get(
                        "duration"
                    ),

                    "detected_classes": video_stats.get(
                        "detected_classes",
                        []
                    )
                }

            return {
                "detected": False
            }

        # ─────────────────────────────
        # UNSUPPORTED FILE TYPE
        # ─────────────────────────────

        logger.warning(
            f"Unsupported file type: {file_ext}"
        )

        return False

    # def is_animal(
    #     self,
    #     source: Union[str, Path, np.ndarray],
    #     quick_mode: bool = True
    # ) -> bool:
    #     """
    #     Robust unified function to check if source contains animals.
    #     Uses temporal consistency, adaptive thresholds, and confidence aggregation.
    #     """

    #     # ─────────────────────────────
    #     # IMAGE (single inference)
    #     # ─────────────────────────────
    #     if isinstance(source, np.ndarray):
    #         result = self.detect_image(source)
    #         return result.has_animal

    #     source_path = Path(source)

    #     if not source_path.exists():
    #         logger.error(f"File not found: {source}")
    #         return False

    #     video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    #     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    #     file_ext = source_path.suffix.lower()

    #     if file_ext in image_extensions:
    #         result = self.detect_image(source)
    #         return result.has_animal

    #     # ─────────────────────────────
    #     # VIDEO (robust decision logic)
    #     # ─────────────────────────────
    #     if file_ext in video_extensions:

    #         frame_skip = 5 if quick_mode else 2
    #         max_frames = 50 if quick_mode else None

    #         video_stats = self.detect_video(
    #             source,
    #             frame_skip=frame_skip,
    #             max_frames=max_frames,
    #             show_progress=False
    #         )

    #         frames_with_animals = video_stats.get("frames_with_animals", 0)
    #         processed_frames = video_stats.get("processed_frames", 0)
    #         avg_confidence = video_stats.get("avg_confidence", 0.0)

    #         # Safety check
    #         if processed_frames == 0:
    #             return False

    #         # ───── Short video handling ─────
    #         # Prevent single-frame false positives
    #         if processed_frames < 10:
    #             return (
    #                 frames_with_animals >= 2 and
    #                 avg_confidence >= 0.4
    #             )

    #         # ───── Normal video handling ─────
    #         min_required_frames = max(2, int(0.1 * processed_frames))
    #         detection_rate = frames_with_animals / processed_frames

    #         return (
    #             frames_with_animals >= min_required_frames and
    #             detection_rate >= 0.10 and
    #             avg_confidence >= 0.4
    #         )

    #     # ─────────────────────────────
    #     # Unsupported file type
    #     # ─────────────────────────────
    #     logger.warning(f"Unsupported file type: {file_ext}")
    #     return False



# ────────────────────────────────────────────────
#  USAGE EXAMPLES
# ────────────────────────────────────────────────

def example_usage():
    """Demonstrate usage of the AnimalDetector class"""
    
    # Initialize detector
    detector = AnimalDetector(
        model_path="yolov8m.pt",
        conf_threshold=0.30,
        iou_threshold=0.45
    )
    
    # Example 1: Simple check if image contains animals
    print("\n=== Example 1: Quick Animal Check ===")
    has_animal = detector.is_animal("uploads/posts/images/dog4.jpeg")
    print(f"Contains animal: {has_animal}")
    
    # Example 2: Detailed image detection
    print("\n=== Example 2: Detailed Image Detection ===")
    result = detector.detect_image("uploads/posts/images/dog4.jpeg")
    print(f"Has animal: {result.has_animal}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Detected: {result.detected_classes}")
    print(f"Bounding boxes: {result.bbox_count}")
    print(f"Processing time: {result.processing_time*1000:.1f}ms")
    
    # Example 3: Image detection with annotation
    print("\n=== Example 3: Image Detection with Visualization ===")
    result, annotated_img = detector.detect_image("uploads/posts/images/dog4.jpeg", return_annotated=True)
    if result.has_animal:
        cv2.imwrite("output_annotated.jpg", annotated_img)
        print("Saved annotated image")
    
    # Example 4: Video detection with frame skipping
    print("\n=== Example 4: Video Detection ===")
    video_stats = detector.detect_video(
        "uploads/das.mp4",
        frame_skip=10,  # Process every 3rd frame
        save_output="output_video.mp4",
        show_progress=True
    )
    print(f"\nVideo Statistics:")
    print(f"  Frames processed: {video_stats['processed_frames']}")
    print(f"  Frames with animals: {video_stats['frames_with_animals']}")
    print(f"  Detection rate: {video_stats['detection_rate']:.1f}%")
    print(f"  Classes found: {video_stats['detected_classes']}")
    print(f"  Processing speed: {video_stats['fps_processed']:.1f} FPS")
    
    # Example 5: Quick video check
    print("\n=== Example 5: Quick Video Check ===")
    has_animal = detector.is_animal("uploads/das.mp4", quick_mode=True)
    print(f"Video contains animal: {has_animal}")
    
    # Example 6: Batch processing
    print("\n=== Example 6: Batch Processing ===")
    files = ["uploads/posts/images/dog4.jpeg", "uploads/das.mp4", "uploads/posts/images/dog3.jpg"]
    for file in files:
        has_animal = detector.is_animal(file)
        print(f"{file}: {'✓ Animal detected' if has_animal else '✗ No animals'}")


if __name__ == "__main__":
    example_usage()