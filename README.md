# anime_projects-
1) # 🎨 ShapeStream 
# Gesture-Controlled Shape Recognition & Media Player
   An interactive computer vision application that recognizes hand-drawn shapes in real-time and plays associated video/audio content. Draw shapes in the air using finger gestures, and the system will automatically recognize them and trigger corresponding multimedia responses.

   ## ✨ Features

- **Real-time Hand Tracking**: Uses MediaPipe for accurate finger and hand landmark detection
- **Gesture-Based Drawing**: Draw shapes in the air using your index finger
- **Shape Recognition**: Automatically detects circles, rectangles, and triangles
- **Smart Shape Detection**: Algorithms to distinguish shapes based on contour analysis
- **Multimedia Integration**: Plays associated video and audio for recognized shapes
- **Visual Feedback**: 
  - Glowing trail effect while drawing
  - Shape outlines with bounding boxes
  - Video playback fitted precisely to drawn shape boundaries
- **Gesture Controls**:
  - ✊ Fist = Clear canvas
  - ☝️ Index finger = Draw mode
  - ✋ Stop gesture = Detect shape
  - Thumb + Index + Middle finger = Play associated video
## 🎮 Demo Shapes & Media

The application comes pre-configured with three shapes:

| Shape | Video | Audio |
|-------|-------|-------|
| Circle | tsukuyomi.mp4 | tsukuyomi.mp3 |
| Rectangle | madara_uchiha.mp4 | madara_uchiha.mp3 |
| Triangle | jarvis.mp4 | jarvis audio.mp3 |
### Prerequisites

Make sure you have Python 3.7+ installed on your system.

## Technical Architecture

### Hand Tracking Pipeline
- **MediaPipe Hands** detects 21 hand landmarks in real-time  
- Finger state detection determines gesture type  
- Exponential smoothing reduces jitter for smooth drawing  

### Shape Detection Algorithm
- Collects points during drawing phase  
- Applies contour approximation (Douglas-Peucker algorithm)  
- Uses circularity metric to distinguish circles from polygons  
- Classifies shapes based on vertex count and geometric properties  

### Media Integration
- Pygame mixer handles audio playback  
- OpenCV `VideoCapture` manages video streams  
- Video frames are resized and masked to fit drawn shape boundaries  
- Bitwise operations blend video with live camera feed  

### Visual Effects
- Gaussian blur creates glowing trail effect  
- Alpha blending for smooth overlays  
- Mask operations for precise video placement

  ### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shapestream.git
cd shapestream
