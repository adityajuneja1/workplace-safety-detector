import cv2
import numpy as np

class FireDetector:
    def __init__(self):
        self.COLOR_FIRE = (0, 0, 255)  # Red
        self.min_fire_area = 500  # Minimum pixel area to consider as fire
    
    def detect_fire(self, frame):
        """Detect fire/flames in frame using color detection"""
        violations = []
        annotated_frame = frame.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire color range (red-orange-yellow)
        # Lower red range
        lower_red1 = np.array([0, 120, 200])
        upper_red1 = np.array([15, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Upper red range
        lower_red2 = np.array([160, 120, 200])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Orange range
        lower_orange = np.array([15, 120, 200])
        upper_orange = np.array([35, 255, 255])
        mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Combine masks
        fire_mask = mask1 | mask2 | mask3
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_fire_area:
                fire_detected = True
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw on frame
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), self.COLOR_FIRE, 2)
                cv2.putText(annotated_frame, "FIRE DETECTED!", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_FIRE, 2)
                
                violations.append({
                    "type": "fire_detected",
                    "bbox": (x, y, x+w, y+h),
                    "area": area,
                    "confidence": min(area / 5000, 1.0)
                })
        
        if fire_detected:
            # Add large warning
            cv2.putText(annotated_frame, "!!! FIRE ALERT !!!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_FIRE, 3)
        
        return annotated_frame, violations
    
    def detect_smoke(self, frame):
        """Detect smoke in frame using color and texture"""
        violations = []
        annotated_frame = frame.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Smoke is typically gray/white with low saturation
        lower_smoke = np.array([0, 0, 150])
        upper_smoke = np.array([180, 50, 255])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 2000:  # Smoke areas tend to be larger
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (smoke tends to rise, so height > width)
                aspect_ratio = h / w if w > 0 else 0
                
                if aspect_ratio > 0.5:
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                    cv2.putText(annotated_frame, "SMOKE?", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        return annotated_frame, violations