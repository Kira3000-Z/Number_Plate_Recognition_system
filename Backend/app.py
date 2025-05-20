import base64
import numpy as np
import cv2
import easyocr
import torch
from ultralytics import YOLO
import time # Import time for potential debugging or timestamping

# Ensure you are importing Flask and request from flask
from flask import Flask, render_template, request
# Ensure you are importing SocketIO and emit from flask_socketio
from flask_socketio import SocketIO, emit # This is the correct import for Flask-SocketIO

# Configuration 
# Set the desired backend for EasyOCR 
EASYOCR_DEVICE = 'cpu'
if torch.cuda.is_available():
    EASYOCR_DEVICE = 'cuda'
    print("CUDA is available. Using GPU for EasyOCR.")
else:
    print("CUDA not available. Using CPU for EasyOCR.")

# Socket.IO Timeout Configuration 
# Setting them very high (e.g., 5 minutes) should prevent timeouts during processing.
SOCKETIO_PING_INTERVAL = 300 # 5 minutes
SOCKETIO_PING_TIMEOUT = 300 # 5 minutes


# Initialize Flask App and Socket.IO
# Use gevent as the async mode for background tasks, as recommended by Flask-SocketIO docs
app = Flask(__name__)
# Allow cross-origin requests for Socket.IO if your frontend is on a different domain/port
# Initialize SocketIO with the Flask app. Specify async_mode='gevent' for background tasks to work correctly with gevent
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_interval=SOCKETIO_PING_INTERVAL, # Set the ping interval
    ping_timeout=SOCKETIO_PING_TIMEOUT   # Set the ping timeout
) # Initialize SocketIO with the Flask app

# Load Models (Global Scope for efficiency)
# Load EasyOCR reader
print(f"Loading EasyOCR with device: {EASYOCR_DEVICE}...")
try:
    # Use 'gpu=True' if EASYOCR_DEVICE is 'cuda', otherwise it defaults to CPU
    reader = easyocr.Reader(['en'], gpu=(EASYOCR_DEVICE == 'cuda'))
    print("EasyOCR loaded successfully.")
except Exception as e:
        print(f"Error loading EasyOCR: {e}")
        # Print a more explicit error message if EasyOCR fails to load
        print("EasyOCR failed to load. Please ensure you have the necessary dependencies (like CUDA if using GPU) and that the 'easyocr' package is correctly installed.")
        reader = None # Set reader to None if loading fails

# Load YOLOv8 model
print("Loading YOLOv8 model...")
try:
    yolo_model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully.")
    # Set YOLO model to evaluation mode 
    yolo_model.eval()
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    # Print a more explicit error message if YOLO fails to load
    print("YOLOv8 model failed to load. Please ensure you have internet access for the first run to download weights, or that the model file ('yolov8n.pt') is in the correct location.")
    yolo_model = None # Set model to None if loading fails


# Utility Functions 

def decode_frame(base64_string):
    """Decodes a base64 string to an OpenCV image."""
    try:
        # Remove the "data:image/jpeg;base64," prefix if present
        if "data:image/jpeg;base64," in base64_string:
             header, encoded = base64_string.split(',', 1)
        else:
             encoded = base64_string # Assume it's just the base64 data

        binary_data = base64.b64decode(encoded)

        # Use Pillow to open the image from binary data
        # Pillow is more robust for various image formats than direct OpenCV decode from buffer
        # Ensure Pillow is imported (can be imported globally or here)
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(binary_data))

        # Convert Pillow image to NumPy array (RGB)
        numpy_image = np.array(image)

        # Convert RGB to BGR for OpenCV
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        return opencv_image

    except Exception as e:
        # Print error but don't necessarily break the loop unless it's critical
        print(f"Error decoding base64 image: {e}")
        return None

def recognize_number_plate_yolo(image, sid):
    """
    Detects and recognizes number plates in an image using YOLOv8 and EasyOCR.
    Emits the recognized plate text back to the client via Socket.IO.
    This function runs in a background task.
    """
    if image is None:
        print("Background task received None image.")
        return # Exit if image is invalid

    if yolo_model is None:
        print("YOLO model not loaded. Cannot perform detection.")
        return

    if reader is None:
        print("EasyOCR reader not loaded. Cannot perform OCR.")
        return

    try:
        # Perform detection using YOLOv8
        # confidence threshold (conf=0.25), suppress output (verbose=False)
        # Pass the image as a list for prediction
        results = yolo_model([image], conf=0.25, verbose=False)

        for r in results:
            # Check if 'boxes' attribute exists and is not None
            if hasattr(r, 'boxes') and r.boxes is not None:
                # Convert bounding box coordinates to numpy array
                boxes = r.boxes.xyxy.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # Crop the detected number plate
                    cropped_plate = image[y1:y2, x1:x2]

                    # Check if the cropped image is valid (not empty)
                    if cropped_plate.shape[0] == 0 or cropped_plate.shape[1] == 0:
                        continue # Skip empty crops

                    # Perform OCR on the cropped plate
                    # reader.readtext returns a list of (bbox, text, confidence) tuples
                    ocr_results = reader.readtext(cropped_plate)

                    plate_text = ""
                    # Concatenate text from all detected lines/words within the plate region
                    for (bbox, text, conf) in ocr_results:
                        # or clean the extracted text (e.g., remove unwanted characters)
                        plate_text += text + " "

                    plate_text = plate_text.strip() # Remove leading/trailing whitespace

                    # If plate text was found, emit it back to the specific client (using sid)
                    if plate_text:
                        print(f"Detected Plate for client {sid}: {plate_text}")
                        # Use socketio.emit from within the background task
                       
                        socketio.emit('number_plate_result', {'plate': plate_text}, room=sid) # type: ignore

    except Exception as e:
        print(f"Error during number plate recognition (in background task): {e}")


# Flask Routes (for serving the HTML file)
@app.route('/')
def index():
    # Renders the index.html file from the 'templates' folder
    return render_template('index.html')

# Socket.IO Event Handlers 
# These handlers are automatically recognized by Flask-SocketIO
@socketio.on('connect')
def handle_connect():
    # request.sid is provided by Flask-SocketIO for the connected client
    print('Client connected:', request.sid) # type: ignore
    # Emit a status message back to the connected client using emit()
    emit('status', {'data': f'Connected to backend'})

@socketio.on('disconnect')
def handle_disconnect():
    # request.sid is available here too
    print('Client disconnected:', request.sid) # type: ignore

@socketio.on('video_frame')
def handle_video_frame(data):
    # This handler receives the video frame from the frontend.
    # start a background task to process it
    # to avoid blocking the main Socket.IO thread.
    # request.sid is available in the event handler context
    client_sid = request.sid # type: ignore
    # print(f"Received video frame from {client_sid}. Starting background task...") 
    frame = decode_frame(data)

    if frame is not None:
        # Start the ANPR process in a background task.
        # Pass the decoded image and the client's session ID (sid)
        # so the background task can emit results back to the correct client.
        socketio.start_background_task(target=recognize_number_plate_yolo, image=frame, sid=client_sid)
    # else:
        # print(f"Failed to decode frame from {client_sid}.") 


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # socketio.run() to run the Flask app with the Socket.IO server.
    # host='0.0.0.0' makes the server accessible externally (use 127.0.0.1 or localhost for local only)
    # port=5000 is the default port
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
