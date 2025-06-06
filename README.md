# Person Counter API

A Flask-based REST API that uses YOLOv8 for real-time person detection and tracking in videos. This application can process uploaded video files and return processed videos with person counting annotations.

## 🎯 Features

- **Video Upload**: Upload video files in multiple formats (MP4, AVI, MOV, MKV)
- **Person Detection**: Uses YOLOv8 (You Only Look Once) for accurate person detection
- **Person Tracking**: Implements centroid-based tracking to count unique individuals
- **Real-time Processing**: Processes videos frame by frame with progress indicators
- **REST API**: Clean RESTful endpoints for easy integration
- **Download Results**: Download processed videos with annotations

## 📋 Requirements

- Python 3.8+
- Flask
- OpenCV (cv2)
- Ultralytics YOLOv8
- NumPy

## 🚀 Installation

1. **Clone or download the project**
   ```bash
   cd c:\Users\HP\Desktop\code
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure YOLOv8 model is downloaded**
   - The `yolov8n.pt` file should be in the project directory
   - If not present, it will be automatically downloaded on first run

## 🏃‍♂️ Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### 1. Check Status
```http
GET /status
```

#### 2. Upload Video
```http
POST /upload
Content-Type: multipart/form-data

Form data:
- video: [video file]
```

**Response:**
```json
{
    "message": "File uploaded successfully",
    "file_id": "unique-file-id",
    "filename": "unique-filename.mp4"
}
```

#### 3. Process Video
```http
POST /process
Content-Type: application/json

{
    "file_id": "unique-file-id"
}
```

**Response:**
```json
{
    "message": "Video processed successfully",
    "output_filename": "processed_filename.mp4",
    "download_url": "/download/processed_filename.mp4"
}
```

#### 4. Download Processed Video
```http
GET /download/<filename>
```

### Example Usage with cURL

```bash
# Upload a video
curl -X POST -F "video=@your_video.mp4" http://localhost:5000/upload

# Process the video (use file_id from upload response)
curl -X POST -H "Content-Type: application/json" \
     -d '{"file_id":"your-file-id"}' \
     http://localhost:5000/process

# Download processed video
curl -O http://localhost:5000/download/processed_your_video.mp4
```

## 🔧 Configuration

### File Upload Settings
- **Supported formats**: MP4, AVI, MOV, MKV
- **Upload folder**: `uploads/`
- **Output folder**: `outputs/`

### Detection Settings
- **Model**: YOLOv8 Nano (yolov8n.pt)
- **Confidence threshold**: 0.5
- **Person class ID**: 0 (COCO dataset)

### Tracking Settings
- **Tracking method**: Centroid-based
- **Distance threshold**: 100 pixels
- **Max disappeared frames**: 10

## 📁 Project Structure

```
c:\Users\HP\Desktop\code\
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── yolov8n.pt         # YOLOv8 model weights
├── uploads/           # Uploaded video files
├── outputs/           # Processed video files
├── venv/             # Virtual environment (optional)
└── README.md         # This file
```

## 🧠 How It Works

1. **Video Upload**: Users upload video files via the `/upload` endpoint
2. **Frame Processing**: Each frame is processed using YOLOv8 for person detection
3. **Person Tracking**: Detected persons are tracked across frames using centroid tracking
4. **Annotation**: Bounding boxes and unique person count are drawn on each frame
5. **Output Generation**: Processed video is saved with annotations
6. **Download**: Users can download the processed video

## 🎨 Output Features

The processed video includes:
- **Green bounding boxes** around detected persons
- **Unique person count** displayed in red text
- **Frame counter** for progress tracking
- **Real-time tracking IDs** for each person

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError for distutils**
   ```bash
   pip install setuptools
   ```

2. **YOLO model not found**
   - Ensure `yolov8n.pt` is in the project directory
   - The model will be downloaded automatically on first run

3. **Video codec issues**
   - Install additional codecs if needed
   - Try converting video to MP4 format

4. **Memory issues with large videos**
   - Consider resizing videos before processing
   - Increase system memory allocation

### Performance Tips

- Use YOLOv8n (nano) for faster processing
- Process shorter video segments for testing
- Consider GPU acceleration for larger projects

## 📊 Performance

- **Detection Speed**: ~30-60 FPS (depends on hardware)
- **Accuracy**: High precision person detection with YOLOv8
- **Memory Usage**: Moderate (depends on video resolution)

## 🔮 Future Enhancements

- [ ] GPU acceleration support
- [ ] Real-time video streaming
- [ ] Multiple object class detection
- [ ] Database integration for analytics
- [ ] Web interface for easier usage
- [ ] Batch processing capabilities

## 📝 License

This project is for educational and research purposes. Please ensure compliance with YOLOv8 and other dependencies' licenses.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

---

**Note**: This application requires adequate computational resources for video processing. Processing time depends on video length, resolution, and hardware capabilities.
