# Workplace Safety Violation Detector

A real-time computer vision system that detects workplace safety violations from camera feeds using YOLOv8, MediaPipe, and OpenCV.

## What It Detects

- No helmet detection (YOLOv8 + HSV color analysis)
- No safety vest detection (YOLOv8 + HSV color analysis)
- Restricted zone intrusion (point-in-polygon detection)
- Fallen person detection (MediaPipe pose estimation)
- Fire/smoke detection (HSV color masking + contour analysis)

## Tech Stack

- **Object Detection:** YOLOv8 (Ultralytics)
- **Pose Estimation:** MediaPipe
- **Image Processing:** OpenCV
- **Web Interface:** Gradio
- **Alerts:** Telegram Bot API

## Features

- Real-time webcam processing at 20-30 FPS
- Image upload analysis
- Configurable restricted zones
- Telegram alerts with cooldown to prevent spam
- JSON logging with violation screenshots
- Statistics dashboard

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python app.py`

## Project Structure

workplace-safety-detector/
├── detectors/
│   ├── safety_gear_detector.py - Helmet and vest detection
│   ├── zone_detector.py - Restricted area monitoring
│   ├── fall_detector.py - Fallen person detection
│   └── fire_detector.py - Fire and smoke detection
├── alerts/
│   └── telegram_alert.py - Telegram notification system
├── helpers/
│   └── logger.py - Violation logging
├── app.py - Main Gradio web interface
├── config.py - Configuration settings
└── requirements.txt - Dependencies

## How It Works

1. Camera feed or uploaded image is processed frame by frame
2. YOLOv8 detects all persons in the frame
3. Each person is checked for safety gear using HSV color analysis on head and torso regions
4. MediaPipe estimates body pose to detect falls based on joint angles
5. HSV masking scans the full frame for fire/smoke signatures
6. Violations trigger Telegram alerts and are logged with screenshots

## License

MIT
