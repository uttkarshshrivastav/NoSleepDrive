🚗💤 NoSleepDrive
Real-time Drowsiness Detection System for Driver Safety
A deep learning-powered system that monitors driver alertness in real-time by detecting eye closure patterns and yawning behavior, helping prevent accidents caused by drowsy driving.

📋 Table of Contents

Overview<br>
Features<br>
Tech Stack<br>
Performance Metrics<br>
Usage<br>
Configuration<br>
Demo<br>
Project Structure<br>
Known Issues<br>
Author<br>


🎯 Overview
NoSleepDrive is an intelligent drowsiness detection system designed to enhance road safety by continuously monitoring drivers for signs of fatigue. The system uses computer vision and deep learning to analyze facial features in real-time, detecting prolonged eye closure and yawning patterns that indicate drowsiness.
Key Highlights:

Real-time processing with web-based interface<br>
Deep learning models trained on specialized datasets<br>
Temporal tracking to reduce false alarms<br>
Session logging and analytics via cloud database<br>
Non-invasive monitoring through webcam<br>

Target Users:

Researchers in computer vision and driver safety<br>
Students exploring ML applications in safety systems<br>
Developers building intelligent transportation solutions<br>
Organizations focused on road safety initiatives<br>


✨ Features
Core Detection Capabilities

👁️ Eye Closure Detection: CNN-based model detects eye states with high accuracy<br>
🥱 Yawn Detection: Specialized model identifies yawning patterns<br>
⏱️ Temporal Tracking: Smart filtering prevents false positives using time-series analysis<br>
🎯 Multi-Signal Fusion: Combines eye and yawn signals for robust drowsiness detection<br>

System Features

🌐 Web Interface: Browser-based access with real-time webcam streaming<br>
💾 Database Logging: Session and event data stored in Supabase cloud database<br>
📊 Session Analytics: Track drowsiness events across multiple monitoring sessions<br>
🧵 Multi-threaded Processing: Concurrent video processing and web server operation<br>
🎨 Visual Feedback: Real-time probability scores and alerts overlaid on video feed<br>
⚙️ Configurable Thresholds: Adjustable sensitivity for different use cases<br>


🛠️ Tech Stack
Machine Learning & Computer Vision<br>

PyTorch - Deep learning framework<br>
scikit learn<br>
OpenCV - Computer vision library<br>
MediaPipe - Face mesh detection and landmark extraction<br>
EfficientNet-B0 - Base architecture for classification models<br>

Backend & Database<br>

Flask - Web framework for serving the application<br>
Python 3.8+ - Core programming language<br>

Frontend

HTML5/JavaScript - Web interface<br>
Canvas API - Webcam frame capture and streaming<br>

Key Libraries

NumPy - Numerical computations<br>
Threading - Concurrent processing<br>
UUID - Session identification<br>


📊 Performance Metrics<br>
Eye Detection Model<br>
Dataset: MRL  Eye Dataset<br>
MetricScoreAccuracy -97.34%
F1-Score- 98%<br>
Yawn Detection Model<br>
Dataset: YawnDD (Yawn Detection Dataset)<br>
MetricScoreAccuracy-98%<br>
System Performance<br>

Processing Speed: ~10 FPS on standard hardware<br>
Latency: Real-time inference with minimal delay<br>
GPU Support: Optional (runs on CPU, faster with GPU)<br>


🚀 Installation
Prerequisites
System Requirements:

Python 3.8 or higher
Webcam (built-in or external)
Internet connection (for database logging)
4GB RAM minimum
GPU optional (improves performance)

Operating System:

Windows 10/11
macOS 10.14+
Linux (Ubuntu 18.04+)

Step-by-Step Setup
1. Clone the Repository
bashgit clone (https://github.com/uttkarshshrivastav/NoSleepDrive)
cd NoSleepDrive
2. Create Virtual Environment (Recommended)
bashpython -m venv venv

# Windows
venv\Scripts\activate


💻 Usage
Running the System
1. Start the Application
bashpython run_live_new.py
You should see:
Loading models...
Models loaded!
✓ Database: Session started - abc-123-def
Flask server started on http://0.0.0.0:5000
2. Access the Web Interface
Open your browser and navigate to:
http://localhost:5000
3. Grant Camera Permission

Click "Allow" when prompted for webcam access
Position yourself facing the camera
Ensure good lighting conditions

4. Monitor Detection
The system will display:

Eye Closure Probability: Real-time score (0.0 - 1.0)
Yawn Probability: Real-time score (0.0 - 1.0)
Eye State: OPEN, CLOSED, or DROWSY
Alerts: Visual warnings when drowsiness detected

5. Stop the System

Press q in the OpenCV window, OR
Press Ctrl+C in the terminal

Session data will be automatically saved to the database.
Viewing Analytics
Option 1: Python Script
bashpython simple_analysis.py
Generates charts:

session_overview.png - Events across sessions
drowsy_event_analysis.png - Event distribution
session_timeline_[id].png - Timeline for specific session


⚙️ Configuration
Adjustable Parameters
Edit run_live.py to customize detection sensitivity:
python# Yawn detection threshold
yawn_threshold = 0.4  # Range: 0.0 - 1.0

# Eye tracking parameters
eye_tracker = EyeTemporalTracker(<br>
    close_th=0.4,           # Threshold for eye closure (0.0 - 1.0)<br>
    open_th=0.3,            # Threshold for eye opening (0.0 - 1.0)<br>
    drowsy_duration=1.5     # Seconds before drowsy alert<br>
)<br>
Model Checkpoint Paths<br>
Update paths in run_live.py if your models are in different locations:<br>
pythonEYE_CKPT = r"path/to/eye_model_checkpoint.pth"<br>
YAWN_CKPT = r"path/to/yawn_model_checkpoint.pth"<br>
Database Configuration<br>
Edit supabase_config.py:<br>
pythonSUPABASE_URL = "your-project-url"<br>
SUPABASE_KEY = "your-anon-key"<br>
To disable database logging, comment out database imports in run_live.py.<br>
Flask Server Settings<br>
Modify web_app_new.py for custom port or host:<br>
pythonapp.run(host="0.0.0.0", port=5000)  # Change port if needed<br>

🎬 Demo
Video Demonstration<br><br>


https://github.com/user-attachments/assets/11663603-0e83-49fe-b4e2-7fc2995891ec




##📁 Project Structure
```text
NoSleepDrive/
│
├── run_live.py                 # Main application entry point
├── web_app_new.py              # Flask web server
├── db_logger.py                # Database operations (Supabase)
├── supabase_config.py          # Database configuration
├── shared_state.py             # Shared state management
├── shared_frame.py             # Shared frame buffer
├── simple_analysis.py          # Analytics & visualization
├── database_schema.sql         # Database schema for Supabase
├── requirements.txt            # Python dependencies
├── check_deployment.py         # System readiness check
│
├── preprocessing/
│   └── roi_extractor.py        # MediaPipe face mesh & ROI extraction
│
├── inference/
│   ├── eye_infer.py            # Eye model inference
│   ├── mouth_infer.py          # Yawn model inference
│   ├── eye_fusion.py           # Eye signal fusion logic
│   └── model_architecture.py   # PyTorch model definitions
│
├── temporal/
│   ├── eye_temporal.py         # Temporal eye state tracking
│   └── mouth_temporal.py       # Yawn event detection
│
├── database/                    # Persistence layer
│   ├── db_logger.py
│   ├── supabase_config.py
│   └── database_schema.sql
├── templates/
│   └── read.html               # Web interface HTML
│
├── training/                    # Model training (offline)
│   ├── eye_model/
│   │   ├── train.py
│   │   └── checkpoints/
│   └── yawn_model/
│       ├── train.py
│       └── checkpoints/
```



⚠️ Known Issues & Limitations
Environmental Constraints

Lighting Sensitivity: System performance degrades in:

Very low light conditions (darkness)
Strong backlighting (light source behind the user)
Recommended: Use frontal lighting for best results
system may have little lag while running on the web site after runing for some time 


System Limitations

Single User: Currently supports one user at a time

Multiple simultaneous users may cause frame conflicts


Internet Dependency: Requires internet for database logging

System continues without logging if database unavailable


Browser Compatibility: Webcam access requires HTTPS or localhost

Use modern browsers (Chrome, Firefox, Edge, Safari)



Performance Considerations

CPU-only processing may result in lower FPS
GPU acceleration recommended for smoother performance
Frame rate depends on hardware capabilities

Workarounds

For low light: Use external lighting or increase ambient brightness
For backlighting: Reposition to avoid light sources behind you
For offline use: Comment out database logging in code


🗺️ Roadmap
Future enhancements being considered:

 Multi-user Support: Simultaneous monitoring of multiple drivers
 Mobile Application: iOS/Android apps for portable monitoring
 Advanced Alerts: SMS/email notifications for drowsiness events
 Head Pose Detection: Additional safety metric for attention tracking
 Audio Alerts: Sound-based warnings for drowsiness
 Dashboard Enhancement: Real-time web dashboard with live charts
 Model Improvements: Continual training with expanded datasets
 Export Features: PDF reports and CSV data export
 Integration APIs: Connect with vehicle systems and IoT devices
 Adaptive Thresholds: Automatic calibration based on user behavior

Note: This is a concept demonstration. Timeline and implementation are subject to change.


👤 Author
Uttkarsh Shrivastav

📧 Email: uttkarsh_s@es.iitr.ac.in 
🐙 GitHub: uttkarshshrivastav


🙏 Acknowledgments
Datasets

MRL Dataset - Eye detection model training
YawnDD Dataset - Yawn detection model training

Technologies

MediaPipe by Google - Face mesh detection
PyTorch - Deep learning framework
Supabase - Cloud database platform
Flask - Web framework

Inspiration
This project was developed to address the critical issue of drowsy driving, which causes thousands of accidents annually. By leveraging modern computer vision and deep learning,
we aim to contribute to road safety technology.

This is a demonstration/research project
Not intended for production deployment in vehicles
Use as supplementary safety tool, not primary safety system
Always prioritize rest over technology for drowsy driving prevention


<div align="center">
⚠️ SAFETY DISCLAIMER ⚠️
This system is a research prototype and should NOT be relied upon as the sole safety measure while driving. If you feel drowsy, the safest action is to stop driving and rest. No technology can replace adequate sleep and alertness.

Built with ❤️ for Road Safety
Drive Safe, Stay Awake, Arrive Alive 🚗💤✨
</div>
