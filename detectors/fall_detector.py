import cv2
import mediapipe as mp
import numpy as np

class FallDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.COLOR_FALLEN = (0, 0, 255)  # Red
        self.COLOR_STANDING = (0, 255, 0)  # Green
        
        # Track previous states for smoothing
        self.fall_history = []
        self.history_size = 5
    
    def get_body_angle(self, landmarks):
        """Calculate body angle from pose landmarks"""
        # Get key points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate midpoints
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        hip_mid = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        
        # Calculate angle from vertical
        torso_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0, -1])
        
        # Angle between torso and vertical
        cos_angle = np.dot(torso_vector, vertical_vector) / (
            np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angle_degrees = np.degrees(angle)
        
        return angle_degrees
    
    def get_body_ratio(self, landmarks):
        """Calculate width to height ratio of body"""
        # Get bounding points
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        if height < 0.01:
            return 0
        
        ratio = width / height
        return ratio
    
    def is_fallen(self, landmarks):
        """Determine if person has fallen based on pose"""
        angle = self.get_body_angle(landmarks)
        ratio = self.get_body_ratio(landmarks)
        
        # Person is likely fallen if:
        # 1. Body angle > 45 degrees from vertical
        # 2. Width/height ratio > 1.5 (body is more horizontal)
        
        fallen = angle > 45 or ratio > 1.5
        
        # Smooth detection with history
        self.fall_history.append(fallen)
        if len(self.fall_history) > self.history_size:
            self.fall_history.pop(0)
        
        # Only trigger if majority of recent frames show fallen
        fall_count = sum(self.fall_history)
        is_fallen_confirmed = fall_count >= (self.history_size // 2 + 1)
        
        return is_fallen_confirmed, angle, ratio
    
    def detect_falls(self, frame):
        """Detect if any person in frame has fallen"""
        violations = []
        annotated_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Check if fallen
            is_fallen, angle, ratio = self.is_fallen(landmarks)
            
            # Draw pose
            self.mp_draw.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            if is_fallen:
                violations.append({
                    "type": "person_fallen",
                    "angle": angle,
                    "ratio": ratio,
                    "confidence": min(angle / 90, 1.0)
                })
                
                # Add warning text
                cv2.putText(annotated_frame, "PERSON FALLEN!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_FALLEN, 3)
                cv2.putText(annotated_frame, f"Angle: {angle:.1f} deg", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_FALLEN, 2)
            else:
                cv2.putText(annotated_frame, "Status: Standing", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_STANDING, 2)
        
        return annotated_frame, violations
    
    def release(self):
        """Release resources"""
        self.pose.close()