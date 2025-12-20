import cv2
import os
import json
from datetime import datetime

class ViolationLogger:
    def __init__(self, recordings_path="recordings"):
        self.recordings_path = recordings_path
        self.violations_log = []
        
        # Create recordings directory if not exists
        if not os.path.exists(recordings_path):
            os.makedirs(recordings_path)
        
        # Create subdirectories for each violation type
        violation_types = ["no_helmet", "no_vest", "restricted_zone", "fire_detected", "person_fallen"]
        for v_type in violation_types:
            type_path = os.path.join(recordings_path, v_type)
            if not os.path.exists(type_path):
                os.makedirs(type_path)
    
    def log_violation(self, violation_type, frame, confidence=0.0, details=None):
        """Log a violation with screenshot"""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save screenshot
        filename = f"{violation_type}_{timestamp_str}.jpg"
        filepath = os.path.join(self.recordings_path, violation_type, filename)
        cv2.imwrite(filepath, frame)
        
        # Create log entry
        log_entry = {
            "type": violation_type,
            "timestamp": timestamp.isoformat(),
            "confidence": confidence,
            "screenshot": filepath,
            "details": details or {}
        }
        
        self.violations_log.append(log_entry)
        
        # Save to JSON file
        self.save_log()
        
        return log_entry
    
    def save_log(self):
        """Save violations log to JSON file"""
        log_file = os.path.join(self.recordings_path, "violations_log.json")
        
        with open(log_file, "w") as f:
            json.dump(self.violations_log, f, indent=2)
    
    def load_log(self):
        """Load violations log from JSON file"""
        log_file = os.path.join(self.recordings_path, "violations_log.json")
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                self.violations_log = json.load(f)
        
        return self.violations_log
    
    def get_stats(self):
        """Get violation statistics"""
        stats = {
            "total": len(self.violations_log),
            "by_type": {},
            "by_hour": {},
            "recent": []
        }
        
        for violation in self.violations_log:
            v_type = violation["type"]
            
            # Count by type
            if v_type not in stats["by_type"]:
                stats["by_type"][v_type] = 0
            stats["by_type"][v_type] += 1
            
            # Count by hour
            try:
                hour = datetime.fromisoformat(violation["timestamp"]).hour
                hour_str = f"{hour:02d}:00"
                if hour_str not in stats["by_hour"]:
                    stats["by_hour"][hour_str] = 0
                stats["by_hour"][hour_str] += 1
            except:
                pass
        
        # Get recent violations (last 10)
        stats["recent"] = self.violations_log[-10:][::-1]
        
        return stats
    
    def clear_log(self):
        """Clear all violations"""
        self.violations_log = []
        self.save_log()