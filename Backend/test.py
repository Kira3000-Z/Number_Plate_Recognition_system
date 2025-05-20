import cv2
import numpy as np
import easyocr
from ultralytics import YOLO # Import YOLO from ultralytics

#  Configuration 
# Set the path to your video file
VIDEO_SOURCE = 0 # Use 0 for default camera

# Confidence threshold for YOLO detections 
CONFIDENCE_THRESHOLD = 0.5


try:
    print("Attempting to load YOLO model...")
    yolo_model = YOLO('yolov8n.pt')
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure you have internet access for the first run to download weights.")
    yolo_model = None # Set to None if loading fails

# Initialize EasyOCR Reader 
print("Initializing EasyOCR reader...")
try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR reader initialized.")
except Exception as e:
    print(f"Error initializing EasyOCR reader: {e}")
    reader = None # Set to None if initialization fails

# Function to detect and recognize number plates using YOLO 
def recognize_number_plate_yolo(frame):
    """
    Processes a single frame to detect number plates using YOLO
    and recognize text using EasyOCR.

    Args:
        frame (np.ndarray): The video frame (image).

    Returns:
        tuple: A tuple containing:
            - list: A list of recognized number plate texts.
            - np.ndarray: The frame with detected bounding boxes and text drawn.
    """
    recognized_plates = []
    frame_with_drawings = frame.copy() # Create a copy to draw on

    if yolo_model is None:
        print("YOLO model not loaded, skipping detection.")
        return recognized_plates, frame_with_drawings

    if reader is None:
        print("EasyOCR reader not initialized, skipping recognition.")
        return recognized_plates, frame_with_drawings


    # YOLO Detection 

    results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD)

    # Process results
    for r in results:
        boxes = r.boxes # Boxes object

        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            # class_name = yolo_model.names[class_id] 

            # Extract the potential number plate region
            plate_roi = frame[y1:y2, x1:x2]

            if plate_roi.shape[0] > 0 and plate_roi.shape[1] > 0: # Ensure ROI is valid
                # OCR using EasyOCR
                try:
                    result = reader.readtext(plate_roi)

               
                    for (bbox, text, prob) in result:
                     
                        recognized_plates.append(text)

                        cv2.rectangle(frame_with_drawings, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_with_drawings, f'{text} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        

                except Exception as e:
                     print(f"Error during EasyOCR readtext: {e}")


    return recognized_plates, frame_with_drawings

#  Main Video Processing Loop
def process_video_feed_yolo():
    """
    Captures video feed, processes frames using YOLO and EasyOCR.
    """
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    print("Processing video feed with YOLO...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Recognize number plates using YOLO and EasyOCR
        recognized_plates, frame_with_drawings = recognize_number_plate_yolo(frame)

        # Output/Communication

        if recognized_plates:
            print(f"Recognized Plates: {recognized_plates}")
            # You would send this data to your frontend 

        # Display the frame with detected plates 
        cv2.imshow('Number Plate Recognition (YOLO)', frame_with_drawings)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

# Run the processing function 
if __name__ == "__main__":
    process_video_feed_yolo()
