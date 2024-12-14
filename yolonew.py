import cv2
import pyttsx3
from time import time
from ultralytics import YOLO

def detect_objects(frame, model, conf_threshold=0.4, last_announcement_time=0, pause_time=15, spoken_objects=set()):
    """
    Detect objects in a single frame using YOLOv8 and provide voice feedback.
    """
    current_time = time()
    detected_objects = set()  # Track objects detected in the current frame

    # Perform detection (direct model call)
    results = model(frame)  # Simplified model call (no need for stream=True)
    detections = results[0].boxes  # Access the boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        label = model.names[class_id]

        if conf > conf_threshold:
            # Draw bounding box around detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # Display object label
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            # Announce object if it hasn't been spoken recently
            if label not in spoken_objects and label not in detected_objects and current_time - last_announcement_time > pause_time:
                detected_objects.add(label)  # Add to current frame's detected objects
                spoken_objects.add(label)  # Add to spoken objects to prevent re-announcement
                engine = pyttsx3.init()
                engine.say(f"Detected: {label}")
                engine.runAndWait()
                last_announcement_time = current_time

    # Display the names of detected objects on the left side of the frame
    y_offset = 30
    for label in detected_objects:
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30  # Space out each name vertically

    return frame, last_announcement_time, spoken_objects

def detect_from_stream(source, model, frame_skip=2):
    """
    Detect objects from an IP camera stream.
    """
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        print("[ERROR] Could not open video stream.")
        return

    print("[INFO] Detecting objects...")
    window_name = "Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # No fixed frame size

    last_announcement_time = 0  # Track the last time an object was announced
    pause_time = 10  # Pause between voice announcements (in seconds)
    frame_count = 0  # Counter to skip frames
    spoken_objects = set()  # Track objects that have been spoken

    while True:
        ret, frame = video.read()
        if not ret:
            print("[INFO] End of stream or failed to grab frame.")
            break

        frame_count += 1
        if frame_count % frame_skip == 0:  # Skip frames based on frame_skip value
            processed_frame, last_announcement_time, spoken_objects = detect_objects(
                frame, model, last_announcement_time=last_announcement_time, pause_time=pause_time, spoken_objects=spoken_objects
            )
            cv2.imshow(window_name, processed_frame)

        if cv2.waitKey(1) == ord("q"):  # Exit on 'q'
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[INFO] Loading YOLOv8 Nano model...")  # Use YOLOv8 Nano model for performance
    model = YOLO("yolov8n.pt")  # Use YOLOv8 Nano pre-trained weights for better performance on low-end devices

    # Specify the IP camera URL here (e.g., http://<IP>:<PORT>/video)
    ip_camera_url = "http://192.168.0.128:8080/video"
    
    print("[INFO] Using IP camera stream.")
    detect_from_stream(ip_camera_url, model, frame_skip=2)  # Skip every 2nd frame for faster processing
