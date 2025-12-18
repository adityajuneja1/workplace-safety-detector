# Detection confidence thresholds
HELMET_CONFIDENCE = 0.5
VEST_CONFIDENCE = 0.5
PERSON_CONFIDENCE = 0.5
FIRE_CONFIDENCE = 0.6

# Alert settings
ALERT_COOLDOWN = 30  # seconds between alerts for same violation
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = "your-bot-token-here"
TELEGRAM_CHAT_ID = "your-chat-id-here"

# Recording settings
SAVE_VIOLATIONS = True
RECORDINGS_PATH = "recordings"

# Zone settings (will be configured in UI)
RESTRICTED_ZONES = []

# Model paths
YOLO_MODEL = "yolov8n.pt"  # Will download automatically

# Violation types
VIOLATIONS = {
    "no_helmet": "No Helmet Detected",
    "no_vest": "No Safety Vest Detected", 
    "restricted_zone": "Person in Restricted Zone",
    "fire_detected": "Fire/Smoke Detected",
    "person_fallen": "Person Fallen Down"
}