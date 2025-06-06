from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
from collections import defaultdict
import tempfile

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 10
        
    def update(self, detections):
        if len(detections) == 0:
            # Mark all existing tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # Simple centroid tracking
        input_centroids = []
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
        
        if len(self.tracks) == 0:
            for centroid in input_centroids:
                self.tracks[self.next_id] = {
                    'centroid': centroid,
                    'disappeared': 0
                }
                self.next_id += 1
        else:
            # Calculate distances and assign
            track_ids = list(self.tracks.keys())
            track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]
            
            # Simple nearest neighbor assignment
            used_rows = set()
            used_cols = set()
            assignments = {}
            
            for i, input_centroid in enumerate(input_centroids):
                min_dist = float('inf')
                best_track = None
                
                for j, track_id in enumerate(track_ids):
                    if j in used_cols:
                        continue
                        
                    track_centroid = track_centroids[j]
                    dist = np.sqrt((input_centroid[0] - track_centroid[0])**2 + 
                                 (input_centroid[1] - track_centroid[1])**2)
                    
                    if dist < min_dist and dist < 100:  # Distance threshold
                        min_dist = dist
                        best_track = track_id
                        best_col = j
                
                if best_track is not None:
                    assignments[i] = best_track
                    used_rows.add(i)
                    used_cols.add(best_col)
            
            # Update existing tracks
            for track_id in track_ids:
                if track_id in assignments.values():
                    input_idx = [k for k, v in assignments.items() if v == track_id][0]
                    self.tracks[track_id]['centroid'] = input_centroids[input_idx]
                    self.tracks[track_id]['disappeared'] = 0
                else:
                    self.tracks[track_id]['disappeared'] += 1
            
            # Add new tracks for unassigned detections
            for i, centroid in enumerate(input_centroids):
                if i not in used_rows:
                    self.tracks[self.next_id] = {
                        'centroid': centroid,
                        'disappeared': 0
                    }
                    self.next_id += 1
            
            # Remove disappeared tracks
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        return list(self.tracks.keys())

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = PersonTracker()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Extract person detections (class 0 is person in COCO)
        person_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if box.cls == 0 and box.conf > 0.5:  # Person class with confidence > 0.5
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        person_detections.append((x1, y1, x2, y2))
        
        # Update tracker
        track_ids = tracker.update(person_detections)
        unique_count = len(track_ids)
        
        # Draw bounding boxes and annotations
        for (x1, y1, x2, y2) in person_detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add unique person count text
        text = f"Unique Persons: {unique_count}"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add frame number
        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:  # Progress indicator
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Video processing complete! Total frames: {frame_count}")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            file_id = str(uuid.uuid4())
            filename = f"{file_id}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            return jsonify({
                'message': 'File uploaded successfully',
                'file_id': file_id,
                'filename': filename
            }), 200
        else:
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video_endpoint():
    try:
        data = request.get_json()
        if not data or 'file_id' not in data:
            return jsonify({'error': 'file_id is required'}), 400
        
        file_id = data['file_id']
        
        # Find the uploaded file
        input_file = None
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.startswith(file_id):
                input_file = filename
                break
        
        if not input_file:
            return jsonify({'error': 'File not found'}), 404
        
        input_path = os.path.join(UPLOAD_FOLDER, input_file)
        output_filename = f"processed_{input_file}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Process the video
        print(f"Starting video processing for {input_file}...")
        process_video(input_path, output_path)
        
        return jsonify({
            'message': 'Video processed successfully',
            'output_filename': output_filename,
            'download_url': f'/download/{output_filename}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'message': 'Person Counter API is running',
        'endpoints': {
            'upload': '/upload - POST with video file',
            'process': '/process - POST with file_id',
            'download': '/download/<filename> - GET processed video'
        }
    })

if __name__ == '__main__':
    print("Starting Person Counter API...")
    print("Endpoints:")
    print("- POST /upload - Upload video file")
    print("- POST /process - Process uploaded video")
    print("- GET /download/<filename> - Download processed video")
    print("- GET /status - API status")
    app.run(debug=True, host='0.0.0.0', port=5000)