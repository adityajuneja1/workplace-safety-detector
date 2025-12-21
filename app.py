import gradio as gr
import cv2
import numpy as np
from datetime import datetime

from detectors.safety_gear_detector import SafetyGearDetector
from detectors.zone_detector import ZoneDetector
from detectors.fall_detector import FallDetector
from detectors.fire_detector import FireDetector
from alerts.telegram_alert import TelegramAlert
from helpers.logger import ViolationLogger
from config import *

# Initialize detectors
safety_detector = SafetyGearDetector()
zone_detector = ZoneDetector()
fall_detector = FallDetector()
fire_detector = FireDetector()

# Initialize logger
logger = ViolationLogger(RECORDINGS_PATH)
logger.load_log()

# Initialize Telegram (disabled by default)
telegram = TelegramAlert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED)

def process_frame(frame, check_helmet, check_vest, check_zone, check_fall, check_fire):
    """Process a single frame for all enabled violations"""
    if frame is None:
        return None, "No frame received"
    
    all_violations = []
    annotated_frame = frame.copy()
    
    # Detect persons first (needed for multiple checks)
    persons = safety_detector.detect_persons(frame)
    
    # Check helmet and vest
    if check_helmet or check_vest:
        annotated_frame, violations = safety_detector.analyze_frame(annotated_frame)
        if check_helmet:
            helmet_violations = [v for v in violations if v["type"] == "no_helmet"]
            all_violations.extend(helmet_violations)
        if check_vest:
            vest_violations = [v for v in violations if v["type"] == "no_vest"]
            all_violations.extend(vest_violations)
    
    # Check restricted zones
    if check_zone and len(zone_detector.restricted_zones) > 0:
        annotated_frame, zone_violations = zone_detector.check_intrusions(persons, annotated_frame)
        all_violations.extend(zone_violations)
    elif check_zone:
        annotated_frame = zone_detector.draw_zones(annotated_frame)
    
    # Check for falls
    if check_fall:
        annotated_frame, fall_violations = fall_detector.detect_falls(annotated_frame)
        all_violations.extend(fall_violations)
    
    # Check for fire
    if check_fire:
        annotated_frame, fire_violations = fire_detector.detect_fire(annotated_frame)
        all_violations.extend(fire_violations)
    
    # Log violations and send alerts
    for violation in all_violations:
        logger.log_violation(
            violation["type"],
            annotated_frame,
            violation.get("confidence", 0.0)
        )
        telegram.send_violation_alert(
            violation["type"],
            annotated_frame
        )
    
    # Create status text
    status = f"Persons detected: {len(persons)}\n"
    status += f"Active violations: {len(all_violations)}\n"
    if all_violations:
        status += "\nViolations:\n"
        for v in all_violations:
            status += f"  - {VIOLATIONS.get(v['type'], v['type'])}\n"
    else:
        status += "\n✅ No violations detected"
    
    return annotated_frame, status

def process_webcam(frame, check_helmet, check_vest, check_zone, check_fall, check_fire):
    """Process webcam frame"""
    if frame is None:
        return None, "Webcam not available"
    return process_frame(frame, check_helmet, check_vest, check_zone, check_fall, check_fire)

def process_image(image, check_helmet, check_vest, check_zone, check_fall, check_fire):
    """Process uploaded image"""
    if image is None:
        return None, "No image uploaded"
    return process_frame(image, check_helmet, check_vest, check_zone, check_fall, check_fire)

def add_zone(image, evt: gr.SelectData):
    """Add point to zone when image is clicked"""
    global zone_points
    if not hasattr(add_zone, 'points'):
        add_zone.points = []
    
    add_zone.points.append([evt.index[0], evt.index[1]])
    
    # Draw points on image
    if image is not None:
        annotated = image.copy()
        for i, point in enumerate(add_zone.points):
            cv2.circle(annotated, tuple(point), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(annotated, tuple(add_zone.points[i-1]), tuple(point), (0, 255, 0), 2)
        if len(add_zone.points) > 2:
            cv2.line(annotated, tuple(add_zone.points[-1]), tuple(add_zone.points[0]), (0, 255, 0), 2)
        return annotated, f"Points: {len(add_zone.points)} (Need at least 3)"
    
    return image, f"Points: {len(add_zone.points)}"

def save_zone():
    """Save the current zone"""
    if hasattr(add_zone, 'points') and len(add_zone.points) >= 3:
        zone_detector.add_zone(add_zone.points)
        add_zone.points = []
        return f"✅ Zone saved! Total zones: {len(zone_detector.restricted_zones)}"
    return "❌ Need at least 3 points to create a zone"

def clear_zones():
    """Clear all zones"""
    zone_detector.clear_zones()
    if hasattr(add_zone, 'points'):
        add_zone.points = []
    return "✅ All zones cleared"

def get_statistics():
    """Get violation statistics"""
    stats = logger.get_stats()
    
    text = f"## 📊 Violation Statistics\n\n"
    text += f"**Total Violations:** {stats['total']}\n\n"
    
    text += "### By Type:\n"
    for v_type, count in stats["by_type"].items():
        text += f"- {VIOLATIONS.get(v_type, v_type)}: {count}\n"
    
    text += "\n### Recent Violations:\n"
    for v in stats["recent"][:5]:
        text += f"- {v['type']} at {v['timestamp'][:19]}\n"
    
    return text

# Create Gradio Interface
with gr.Blocks(title="Workplace Safety Detector", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🦺 Workplace Safety Violation Detector")
    gr.Markdown("Real-time detection of safety violations: No helmet, No vest, Zone intrusion, Falls, Fire")
    
    with gr.Tabs():
        # Tab 1: Live Webcam
        with gr.TabItem("📹 Live Webcam"):
            with gr.Row():
                with gr.Column():
                    webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
                    with gr.Row():
                        helmet_check = gr.Checkbox(value=True, label="🪖 Helmet Detection")
                        vest_check = gr.Checkbox(value=True, label="🦺 Vest Detection")
                    with gr.Row():
                        zone_check = gr.Checkbox(value=False, label="⛔ Zone Intrusion")
                        fall_check = gr.Checkbox(value=True, label="🚨 Fall Detection")
                        fire_check = gr.Checkbox(value=True, label="🔥 Fire Detection")
                
                with gr.Column():
                    output_image = gr.Image(label="Detection Result")
                    status_text = gr.Textbox(label="Status", lines=6)
            
            webcam.stream(
                process_webcam,
                inputs=[webcam, helmet_check, vest_check, zone_check, fall_check, fire_check],
                outputs=[output_image, status_text]
            )
        
        # Tab 2: Image Upload
        with gr.TabItem("🖼️ Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_image = gr.Image(label="Upload Image", type="numpy")
                    with gr.Row():
                        img_helmet = gr.Checkbox(value=True, label="🪖 Helmet")
                        img_vest = gr.Checkbox(value=True, label="🦺 Vest")
                    with gr.Row():
                        img_zone = gr.Checkbox(value=False, label="⛔ Zone")
                        img_fall = gr.Checkbox(value=True, label="🚨 Fall")
                        img_fire = gr.Checkbox(value=True, label="🔥 Fire")
                    analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                
                with gr.Column():
                    img_output = gr.Image(label="Detection Result")
                    img_status = gr.Textbox(label="Status", lines=6)
            
            analyze_btn.click(
                process_image,
                inputs=[upload_image, img_helmet, img_vest, img_zone, img_fall, img_fire],
                outputs=[img_output, img_status]
            )
        
        # Tab 3: Zone Configuration
        with gr.TabItem("⛔ Configure Zones"):
            gr.Markdown("### Draw Restricted Zones")
            gr.Markdown("Click on the image to add points. Need at least 3 points to create a zone.")
            
            with gr.Row():
                with gr.Column():
                    zone_image = gr.Image(label="Click to add zone points", type="numpy")
                    with gr.Row():
                        save_zone_btn = gr.Button("💾 Save Zone", variant="primary")
                        clear_zone_btn = gr.Button("🗑️ Clear All Zones", variant="secondary")
                
                with gr.Column():
                    zone_status = gr.Textbox(label="Zone Status", lines=4)
            
            zone_image.select(add_zone, inputs=[zone_image], outputs=[zone_image, zone_status])
            save_zone_btn.click(save_zone, outputs=[zone_status])
            clear_zone_btn.click(clear_zones, outputs=[zone_status])
        
        # Tab 4: Statistics
        with gr.TabItem("📊 Statistics"):
            stats_display = gr.Markdown("Click refresh to see statistics")
            refresh_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
            refresh_btn.click(get_statistics, outputs=[stats_display])
        
        # Tab 5: Settings
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### Telegram Alerts")
            gr.Markdown("Configure Telegram bot for real-time alerts")
            
            tg_token = gr.Textbox(label="Bot Token", type="password")
            tg_chat = gr.Textbox(label="Chat ID")
            tg_enable = gr.Checkbox(label="Enable Telegram Alerts", value=False)
            
            def update_telegram(token, chat_id, enabled):
                global telegram
                telegram = TelegramAlert(token, chat_id, enabled)
                if enabled:
                    success, msg = telegram.test_connection()
                    return f"{'✅' if success else '❌'} {msg}"
                return "Telegram alerts disabled"
            
            save_tg_btn = gr.Button("💾 Save Settings", variant="primary")
            tg_status = gr.Textbox(label="Status")
            save_tg_btn.click(update_telegram, inputs=[tg_token, tg_chat, tg_enable], outputs=[tg_status])

# Launch app
if __name__ == "__main__":
    app.launch()