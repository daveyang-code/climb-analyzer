# Climbing Motion Analyzer

This tool analyzes climbing movements by detecting climbing holds and tracking body positions. It provides visual feedback on hold usage, body positioning, and stability assessment.

# Features

- Hold Detection: Uses YOLO model to detect climbing holds

- Body Tracking: MediaPipe for pose estimation

- Hold-Body Interaction: Tracks which holds are being used by hands/feet

- Stability Analysis: Evaluates climber's center of mass relative to support points

- Visual Feedback: Annotated video output with tracking information

# Dependencies
Make sure you have the following installed:
```
pip install opencv-python numpy mediapipe ultralytics
```

# Images
![climb](https://github.com/user-attachments/assets/e3b6c8ce-a1b8-4d70-8729-3e90d7c2ebf0)
