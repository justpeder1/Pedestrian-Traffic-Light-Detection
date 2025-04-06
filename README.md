# Pedestrian Traffic Light Detection

## Project Overview
This project provides an advanced AI-powered system for detecting and analyzing pedestrians and traffic lights in images and videos. The system uses deep learning techniques to identify various traffic elements in real-time.

## Features
- **Image Detection**: Upload a single image to detect pedestrians and traffic lights
- **Video Detection**: Process video files for detection in motion
- **Real-time Detection**: Perform live detection and analysis using a camera feed

## Detection Capabilities
The system can detect the following elements:
- Red Traffic Lights
- Green Traffic Lights

![Image](https://github.com/user-attachments/assets/13c9738c-5d1a-4159-92f6-b94ae3b71e85)
![Image](https://github.com/user-attachments/assets/a00a72fc-1130-4323-9d68-8b0c0578cccc)
![Image](https://github.com/user-attachments/assets/8bb73f3b-f704-48e3-af3e-5f1ee2fd3269)

## Technical Details
- Built with Flask web framework
- Uses YOLOv8x object detection model
- Trained on a comprehensive dataset for high accuracy
- Supports CUDA acceleration for faster processing
- Includes WebM video processing capabilities

## System Requirements
- Python 3.6+
- PyTorch
- CUDA-enabled GPU (optional but recommended for better performance)
- Flask
- OpenCV

## Installation
1. Clone this repository
2. Download the model file (pedestrianlight.pt)
3. Install requirements:
   ```bash
   pip install torch flask opencv-python ultralytics
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Usage
After starting the application, navigate to:
- Homepage: `http://localhost:5000/`
- Image Detection: `http://localhost:5000/image`
- Video Detection: `http://localhost:5000/video`
- Real-time Detection: `http://localhost:5000/realtime`

## Web Interface
The project includes a modern and user-friendly web interface with:
- Clean navigation sidebar
- Intuitive upload interfaces
- Real-time processing feedback
- Download capability for processed videos and images

## Model Information
The detection model (pedestrianlight.pt) is a YOLOv11-based model specifically trained to identify pedestrians and traffic light states with high accuracy in various lighting and environmental conditions.

## Project Structure
```
├── app.py                   # Main Flask application
├── pedestrianlight.pt       # Pre-trained YOLO model
├── templates/               # HTML templates for web interface
│   ├── home.html            # Homepage template
│   ├── image.html           # Image detection page
│   ├── video.html           # Video detection page
│   └── realtime.html        # Real-time detection page
└── videodetect/             # Output directory for processed videos
``` 
