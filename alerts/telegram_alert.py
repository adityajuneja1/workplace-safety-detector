import requests
import cv2
import os
import time
from datetime import datetime

class TelegramAlert:
    def __init__(self, bot_token, chat_id, enabled=False):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Cooldown tracking to avoid spam
        self.last_alert_time = {}
        self.cooldown_seconds = 30
    
    def can_send_alert(self, violation_type):
        """Check if enough time has passed since last alert of this type"""
        current_time = time.time()
        
        if violation_type not in self.last_alert_time:
            return True
        
        time_since_last = current_time - self.last_alert_time[violation_type]
        return time_since_last >= self.cooldown_seconds
    
    def send_text(self, message):
        """Send a text message via Telegram"""
        if not self.enabled:
            print(f"[ALERT - Not Sent] {message}")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")
            return False
    
    def send_photo(self, frame, caption):
        """Send a photo with caption via Telegram"""
        if not self.enabled:
            print(f"[ALERT - Not Sent] {caption}")
            return False
        
        try:
            # Encode frame to jpg
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            url = f"{self.base_url}/sendPhoto"
            files = {
                "photo": ("violation.jpg", img_encoded.tobytes(), "image/jpeg")
            }
            data = {
                "chat_id": self.chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }
            response = requests.post(url, files=files, data=data, timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Telegram photo: {e}")
            return False
    
    def send_violation_alert(self, violation_type, frame, details=""):
        """Send a violation alert with photo"""
        if not self.can_send_alert(violation_type):
            return False
        
        # Update last alert time
        self.last_alert_time[violation_type] = time.time()
        
        # Create alert message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        violation_names = {
            "no_helmet": "🔴 NO HELMET DETECTED",
            "no_vest": "🟠 NO SAFETY VEST DETECTED",
            "restricted_zone": "⛔ RESTRICTED ZONE INTRUSION",
            "fire_detected": "🔥 FIRE/SMOKE DETECTED",
            "person_fallen": "🚨 PERSON FALLEN DOWN"
        }
        
        alert_name = violation_names.get(violation_type, violation_type.upper())
        
        caption = f"""
<b>⚠️ SAFETY VIOLATION ALERT</b>

<b>Type:</b> {alert_name}
<b>Time:</b> {timestamp}
<b>Details:</b> {details}

<i>Workplace Safety Monitoring System</i>
"""
        
        success = self.send_photo(frame, caption)
        
        if success:
            print(f"[ALERT SENT] {violation_type} at {timestamp}")
        
        return success
    
    def test_connection(self):
        """Test if Telegram bot is working"""
        if not self.enabled:
            return False, "Telegram alerts disabled"
        
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                bot_name = bot_info.get("result", {}).get("username", "Unknown")
                return True, f"Connected to bot: @{bot_name}"
            else:
                return False, "Invalid bot token"
        except Exception as e:
            return False, f"Connection failed: {e}"