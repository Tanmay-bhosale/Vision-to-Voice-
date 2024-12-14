import cv2
import argparse
import pyttsx3
from time import time
from ultralytics import YOLO

def detect_objects(frame, model, conf_threshold=0.4, last_announcement_time=0, pause_time=15):
    """
    Detect objects in a single frame using YOLOv8 and provide voice feedback.
    """
    total_objects = 0
    spoken_objects = set()  # Track objects that have been spoken
    current_time = time()

    # Perform detection (stream=True returns a generator)
    for result in model(frame, stream=True):  # Iterate over generator
        detections = result.boxes
        total_objects += len(detections)  # Count total detections

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            label = model.names[class_id]

            if conf > conf_threshold:
                # Reduce the margin of the rectangle by decreasing the thickness
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # Set a smaller and clearer font for the object label
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                )

                # Announce the object if it hasn't been spoken recently
                if label not in spoken_objects and current_time - last_announcement_time > pause_time:
                    spoken_objects.add(label)
                    engine = pyttsx3.init()
                    engine.say(f"Detected: {label}")
                    engine.runAndWait()
                    last_announcement_time = current_time

    # Display total object count
    cv2.putText(frame, f"Total Objects: {total_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, last_announcement_time

def detect_from_stream(source, model, frame_skip=2):
    """
    Detect objects from a video source (webcam, video file, or IP camera).
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

    while True:
        ret, frame = video.read()
        if not ret:
            print("[INFO] End of stream or failed to grab frame.")
            break

        frame_count += 1
        if frame_count % frame_skip == 0:  # Skip frames based on frame_skip value
            processed_frame, last_announcement_time = detect_objects(frame, model, last_announcement_time=last_announcement_time, pause_time=pause_time)
            cv2.imshow(window_name, processed_frame)

        if cv2.waitKey(1) == ord("q"):  # Exit on 'q'
            break

    video.release()
    cv2.destroyAllWindows()

def detect_from_image(image_path, model):
    """
    Detect objects in a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Could not read the image. Check the path.")
        return

    processed_image, _ = detect_objects(image, model, conf_threshold=0.5, last_announcement_time=0, pause_time=10)
    cv2.imshow("Object Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def args_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv8")
    parser.add_argument("-i", "--image", type=str, help="Path to an image file")
    parser.add_argument("-v", "--video", type=str, help="Path to a video file")
    parser.add_argument("-c", "--camera", action="store_true", help="Use the local webcam")
    parser.add_argument(
        "-ip", "--ip_camera", type=str, help="URL for IP camera stream (e.g., http://<IP>:<PORT>/video)"
    )
    return vars(parser.parse_args())

if __name__ == "__main__":
    print("[INFO] Loading YOLOv8 Nano model...")  # Use YOLOv8 Nano model for performance
    model = YOLO("yolov8n.pt")  # Use YOLOv8 Nano pre-trained weights for better performance on low-end devices

    args = args_parser()

    if args["ip_camera"]:
        print("[INFO] Using IP camera stream.")
        detect_from_stream(args["ip_camera"], model, frame_skip=5)  # Skip every 5th frame
    elif args["camera"]:
        print("[INFO] Using local webcam.")
        detect_from_stream(0, model, frame_skip=5)  # Skip every 5th frame
    elif args["video"]:
        print("[INFO] Using video file.")
        detect_from_stream(args["video"], model, frame_skip=5)  # Skip every 5th frame
    elif args["image"]:
        print("[INFO] Using image file.")
        detect_from_image(args["image"], model)
    else:
        print("[ERROR] No input source specified. Use -h for help.")
