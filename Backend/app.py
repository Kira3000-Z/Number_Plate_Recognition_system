import base64
import numpy as np
import cv2
import easyocr
import torch
from ultralytics import YOLO
import time 
import json
import logging
import torch.serialization
from PIL import Image 
import io 

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit 
from gevent import monkey 
monkey.patch_all() 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Flask App Setup 
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

print("Flask app initialized.")

# SocketIO Setup 
# Initialize SocketIO with the Flask app.

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_interval=300, # 5 minutes (in seconds)
    ping_timeout=300   # 5 minutes (in seconds)
)

print("SocketIO initialized.")

# Configuration 
# Set the desired backend for EasyOCR 
EASYOCR_DEVICE = 'cpu'
if torch.cuda.is_available():
    EASYOCR_DEVICE = 'cuda'
    print("CUDA is available. Using GPU for EasyOCR.")
else:
    print("CUDA not available. Using CPU for EasyOCR.")

# Repeated Detection Filtering Configuration
REQUIRED_CONSECUTIVE_DETECTIONS = 3 # Number of times a plate must be detected consecutively to be sent
DETECTION_HISTORY_SIZE = 5 # How many recent detections to keep track of (optional, for more complex logic)

# Model Loading 
# Declare yolo_model and reader at the top level
yolo_model = None
reader = None

# Attempt to import necessary ultralytics components and initialize variables for Pylance
tasks = None
_DETECTION_MODEL_AVAILABLE = False
try:
    from ultralytics.nn import tasks
    _DETECTION_MODEL_AVAILABLE = hasattr(tasks, 'DetectionModel') and tasks.DetectionModel is not None
    if _DETECTION_MODEL_AVAILABLE:
        app.logger.info("ultralytics.nn.tasks.DetectionModel found.")
    else:
         app.logger.warning("ultralytics.nn.tasks.DetectionModel not found or is None.")
except ImportError:
    _DETECTION_MODEL_AVAILABLE = False
    app.logger.warning("Could not import ultralytics.nn.tasks.")

yolo_models_module = None
_YOLO_MODEL_CLASS_AVAILABLE = False
try:
    import ultralytics.models.yolo.model as yolo_models_module
    _YOLO_MODEL_CLASS_AVAILABLE = hasattr(yolo_models_module, 'Model') and yolo_models_module.Model is not None
    if _YOLO_MODEL_CLASS_AVAILABLE:
         app.logger.info("ultralytics.models.yolo.model.Model found.")
    else:
         app.logger.warning("ultralytics.models.yolo.model.Model not found or is None.")
except ImportError:
    _YOLO_MODEL_CLASS_AVAILABLE = False
    app.logger.warning("Could not import ultralytics.models.yolo.model.")


try:
    app.logger.info("Attempting to load YOLO model...")
    # Build the list of allowed globals dynamically
    allowed_globals = []
    if _DETECTION_MODEL_AVAILABLE and tasks is not None and tasks.DetectionModel is not None:
        allowed_globals.append(tasks.DetectionModel)
        app.logger.info("Added tasks.DetectionModel to allowed_globals.")
    if _YOLO_MODEL_CLASS_AVAILABLE and yolo_models_module is not None and yolo_models_module.Model is not None:
        allowed_globals.append(yolo_models_module.Model)
        app.logger.info("Added yolo_models.Model to allowed_globals.")

    with torch.serialization.safe_globals(allowed_globals if allowed_globals else []):
        # Use yolov8n.pt directly with ultralytics.YOLO
        yolo_model = YOLO('yolov8n.pt')

    if yolo_model is not None:
        yolo_model.eval()
        app.logger.info("YOLO model loaded successfully.")
    else:
        app.logger.warning("YOLO model loading returned None.")
except Exception as e:
    app.logger.error(f"Error loading YOLO model: {e}")
    app.logger.warning("YOLO model will not be available.")

try:
    app.logger.info("Initializing EasyOCR reader...")
    # Use 'gpu=True' if EASYOCR_DEVICE is 'cuda', otherwise it defaults to CPU
    reader = easyocr.Reader(['en'], gpu=(EASYOCR_DEVICE == 'cuda'))
    app.logger.info("EasyOCR reader initialized.")
except Exception as e:
    app.logger.error(f"Error initializing EasyOCR reader: {e}")
    reader = None

# Global state for tracking detections per client
# Use a dictionary to store detection history and counters for each connected client 
client_detection_state = {}

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
    Processes a single frame to detect number plates using YOLO
    and recognize text using EasyOCR.

    Args:
        image (np.ndarray): The video frame (image).
        sid (str): The Socket.IO session ID of the client.

    Emits the recognized plate text back to the client via Socket.IO
    after applying repeated detection filtering.
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
        # Pass the image as a list for prediction
        results = yolo_model([image], conf=0.25, verbose=False)

        recognized_plates_in_frame = []

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
                    x2 = min(w, w) 
                    y2 = min(h, y2)

                    # Crop the detected number plate
                    cropped_plate = image[y1:y2, x1:x2]

                    # Check if the cropped image is valid 
                    if cropped_plate.shape[0] == 0 or cropped_plate.shape[1] == 0:
                        continue # Skip empty crops

                    # Perform OCR on the cropped plate
                    ocr_results = reader.readtext(cropped_plate)

                    plate_text = ""
                    # Concatenate text from all detected lines/words within the plate region
                    for (bbox, text, conf) in ocr_results:
                        plate_text += text + " "

                    plate_text = plate_text.strip() # Remove leading/trailing whitespace

                    # If plate text was found, add it to the list for this frame
                    if plate_text:
                        recognized_plates_in_frame.append(plate_text)

        # Apply Repeated Detection Filtering 
        # Get the detection state for this client
        state = client_detection_state.get(sid)
        if state is None:
             # Should not happen if connect handler works, but good practice
             print(f"State not found for client {sid}. Re-initializing.")
             client_detection_state[sid] = {
                'last_detected_plate': None,
                'consecutive_count': 0,
                'last_sent_plate': None 
            }
             state = client_detection_state[sid]

        # Simple filtering logic: only send if the same plate is detected consecutively
        if recognized_plates_in_frame:
            # Assuming we only care about the first detected plate for simplicity
            current_plate = recognized_plates_in_frame[0]

            if current_plate == state['last_detected_plate']:
                state['consecutive_count'] += 1
                # If consecutive count reaches the threshold, send the plate
                if state['consecutive_count'] >= REQUIRED_CONSECUTIVE_DETECTIONS:
                    # Only send if this plate hasn't just been sent after meeting the threshold
                    if state['last_sent_plate'] != current_plate:
                        print(f"Detected {current_plate} consecutively {state['consecutive_count']} times for client {sid}. Sending.")
                        socketio.emit('number_plate_result', {'plate': current_plate}, room=sid) # type: ignore
                        state['last_sent_plate'] = current_plate # Record the plate that was just sent
            else:
                # New plate detected
                state['last_detected_plate'] = current_plate
                state['consecutive_count'] = 1 # Reset consecutive count
                # state['last_sent_plate'] = None # Don't reset last_sent_plate here

        else:
            # No plate detected in this frame
            state['last_detected_plate'] = None
            state['consecutive_count'] = 0 # Reset consecutive count
            # state['last_sent_plate'] = None # Don't reset last_sent_plate here

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
    """Handler for new SocketIO connections."""
    client_sid = request.sid # type: ignore
    print(f'Client connected: {client_sid}')
    # Initialize state for the new client
    client_detection_state[client_sid] = {
        'last_detected_plate': None,
        'consecutive_count': 0,
        'last_sent_plate': None # Initialize last_sent_plate
    }
    emit('status', {'data': f'Connected to backend'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handler for SocketIO disconnections."""
    client_sid = request.sid # type: ignore
    print(f'Client disconnected: {client_sid}')
    # Clean up state for the disconnected client
    if client_sid in client_detection_state:
        del client_detection_state[client_sid]

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handler for receiving video frames from the frontend.
    We immediately start a background task to process it
    to avoid blocking the main Socket.IO thread.
    """
    client_sid = request.sid # type: ignore
    # print(f"Received video frame from {client_sid}. Starting background task...") # Uncomment for debugging
    frame = decode_frame(data)

    if frame is not None:
        # Start the ANPR process in a background task.
        # Pass the decoded image and the client's session ID (sid)
        # so the background task can emit results back to the correct client.
        socketio.start_background_task(target=recognize_number_plate_yolo, image=frame, sid=client_sid)
    # else:
        # print(f"Failed to decode frame from {client_sid}.") # Uncomment for debugging decode failures

@socketio.on('clear_results')
def handle_clear_results():
    """Handler for receiving a clear results command from the frontend."""
    client_sid = request.sid # type: ignore
    print(f'Received clear results command from {client_sid}')
    # Reset the detection state for this client
    if client_sid in client_detection_state:
         client_detection_state[client_sid] = {
            'last_detected_plate': None,
            'consecutive_count': 0,
            'last_sent_plate': None # Reset last sent plate on clear
        }
    # Emit an event back to the frontend to signal that results should be cleared
    emit('clear_display', room=client_sid) # type: ignore


# --- Main Execution ---
if __name__ == '__main__':
    print("Entering __main__ block to start server.")
    print("Starting Flask-SocketIO server...")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
