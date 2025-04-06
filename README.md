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

![Image](https://github.com/user-attachments/assets/514cd3d6-e572-4589-84b1-e60b7c935cca)
![Image](https://github.com/user-attachments/assets/2050e56c-4f14-4351-a499-91afa051e1c2)
![Image](https://github.com/user-attachments/assets/ecb982bb-d637-4511-ba72-2b4383102345)

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