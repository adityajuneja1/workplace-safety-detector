from ultralytics import YOLO
import cv2
import numpy as np

class SafetyGearDetector:
    def __init__(self):
        # Load YOLOv8 model (downloads automatically)
        self.model = YOLO("yolov8n.pt")
        
        # COCO class IDs we care about
        self.PERSON_CLASS = 0
        
        # Colors for drawing
        self.COLOR_SAFE = (0, 255, 0)      # Green
        self.COLOR_VIOLATION = (0, 0, 255)  # Red
        self.COLOR_WARNING = (0, 165, 255)  # Orange
    
    def detect_persons(self, frame):
        """Detect all persons in the frame"""
        results = self.model(frame, verbose=False)
        persons = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == self.PERSON_CLASS and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf
                    })
        
        return persons
    
    def check_helmet(self, frame, person_bbox):
        """Check if person is wearing helmet by analyzing head region"""
        x1, y1, x2, y2 = person_bbox
        
        # Get head region (top 25% of person bbox)
        head_height = int((y2 - y1) * 0.25)
        head_region = frame[y1:y1+head_height, x1:x2]
        
        if head_region.size == 0:
            return True, 0.0  # Can't determine, assume safe
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Common helmet colors: yellow, orange, white, red
        # Yellow helmet
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # White helmet
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Orange helmet
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([20, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Combine masks
        helmet_mask = yellow_mask | white_mask | orange_mask
        helmet_pixels = cv2.countNonZero(helmet_mask)
        total_pixels = head_region.shape[0] * head_region.shape[1]
        
        if total_pixels == 0:
            return True, 0.0
        
        helmet_ratio = helmet_pixels / total_pixels
        has_helmet = helmet_ratio > 0.15
        
        return has_helmet, helmet_ratio
    
    def check_vest(self, frame, person_bbox):
        """Check if person is wearing safety vest by analyzing torso region"""
        x1, y1, x2, y2 = person_bbox
        
        # Get torso region (25% to 60% of person height)
        torso_top = int(y1 + (y2 - y1) * 0.25)
        torso_bottom = int(y1 + (y2 - y1) * 0.60)
        torso_region = frame[torso_top:torso_bottom, x1:x2]
        
        if torso_region.size == 0:
            return True, 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # High-visibility vest colors: bright yellow, bright orange
        # Bright yellow/green vest
        yellow_lower = np.array([25, 100, 100])
        yellow_upper = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Bright orange vest
        orange_lower = np.array([10, 150, 150])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Combine masks
        vest_mask = yellow_mask | orange_mask
        vest_pixels = cv2.countNonZero(vest_mask)
        total_pixels = torso_region.shape[0] * torso_region.shape[1]
        
        if total_pixels == 0:
            return True, 0.0
        
        vest_ratio = vest_pixels / total_pixels
        has_vest = vest_ratio > 0.20
        
        return has_vest, vest_ratio
    
    def analyze_frame(self, frame):
        """Analyze frame for safety violations"""
        violations = []
        annotated_frame = frame.copy()
        
        # Detect all persons
        persons = self.detect_persons(frame)
        
        for person in persons:
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Check for helmet
            has_helmet, helmet_conf = self.check_helmet(frame, bbox)
            
            # Check for vest
            has_vest, vest_conf = self.check_vest(frame, bbox)
            
            # Determine violations
            person_violations = []
            
            if not has_helmet:
                person_violations.append("no_helmet")
                violations.append({
                    "type": "no_helmet",
                    "bbox": bbox,
                    "confidence": 1 - helmet_conf
                })
            
            if not has_vest:
                person_violations.append("no_vest")
                violations.append({
                    "type": "no_vest",
                    "bbox": bbox,
                    "confidence": 1 - vest_conf
                })
            
            # Draw bounding box
            if person_violations:
                color = self.COLOR_VIOLATION
                label = "VIOLATION: " + ", ".join(person_violations)
            else:
                color = self.COLOR_SAFE
                label = "SAFE"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame, violations