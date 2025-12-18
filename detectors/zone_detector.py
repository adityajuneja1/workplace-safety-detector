import cv2
import numpy as np

class ZoneDetector:
    def __init__(self):
        self.restricted_zones = []
        self.COLOR_ZONE = (0, 0, 255)  # Red
        self.COLOR_INTRUSION = (0, 0, 255)  # Red
    
    def add_zone(self, points):
        """Add a restricted zone defined by polygon points"""
        if len(points) >= 3:
            self.restricted_zones.append(np.array(points, dtype=np.int32))
    
    def clear_zones(self):
        """Clear all restricted zones"""
        self.restricted_zones = []
    
    def get_person_center(self, bbox):
        """Get the bottom center of person bbox (feet position)"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        feet_y = y2  # Bottom of bbox
        return (center_x, feet_y)
    
    def is_in_zone(self, point, zone):
        """Check if a point is inside a polygon zone"""
        result = cv2.pointPolygonTest(zone, point, False)
        return result >= 0
    
    def check_intrusions(self, persons, frame):
        """Check if any person is in a restricted zone"""
        violations = []
        annotated_frame = frame.copy()
        
        # Draw all restricted zones
        for zone in self.restricted_zones:
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.polylines(annotated_frame, [zone], True, self.COLOR_ZONE, 2)
            
            # Add "RESTRICTED" label
            x, y = zone[0]
            cv2.putText(annotated_frame, "RESTRICTED ZONE", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_ZONE, 2)
        
        # Check each person
        for person in persons:
            bbox = person["bbox"]
            feet_pos = self.get_person_center(bbox)
            
            for zone in self.restricted_zones:
                if self.is_in_zone(feet_pos, zone):
                    violations.append({
                        "type": "restricted_zone",
                        "bbox": bbox,
                        "zone": zone.tolist(),
                        "confidence": 1.0
                    })
                    
                    # Draw alert on person
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                 self.COLOR_INTRUSION, 3)
                    cv2.putText(annotated_frame, "ZONE INTRUSION!", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_INTRUSION, 2)
                    break
        
        return annotated_frame, violations
    
    def draw_zones(self, frame):
        """Draw all restricted zones on frame"""
        annotated_frame = frame.copy()
        
        for zone in self.restricted_zones:
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.polylines(annotated_frame, [zone], True, self.COLOR_ZONE, 2)
        
        return annotated_frame