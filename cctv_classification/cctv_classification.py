import cv2  # For image processing
import os  # For file operations
import logging  # For logging
from shutil import move, copy  # For moving files
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import pandas as pd
import numpy as np
import json  # For loading the configuration file
import time  # For time operations
from PIL import Image  # For image verification

COLOR = (255, 0, 0)  # Green color for bounding boxes
THICKNESS = 2  # Thickness of the bounding box lines


def is_image_corrupted(cv2_image):
    if cv2_image is None or cv2_image.size == 0:
        return True

    try:
        Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)).verify()
        return False
    except (ValueError, SyntaxError) as e:
        logging.error(f"Error verifying image: {e}")
        return True


def load_config(config_path="config.json"):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


try:
    config = load_config()  # Load the configuration file
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    config = {}  # Use an empty dictionary if the configuration file could not be loaded


# Centralized configuration class
class Config:
    LOG_LEVEL = (
        logging.INFO
    )  # Different log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # Paths to the TensorFlow object detection model files
    # https://github.com/ChiekoN/OpenCV_SSD_MobileNet/tree/master/model
    MODEL_PATH = "cctv_classification/frozen_inference_graph.pb"
    CONFIG_PATH = "cctv_classification/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

    ## use configuration from config.json
    # CAMERAS = [
    #    f"http://{config['CAMERA1_IP']}/mjpeg.cgi?user={config['CAMERA_USER']}?password={config['CAMERA_PASSWORD']}",
    #    f"http://{config['CAMERA2_IP']}/mjpeg.cgi?user={config['CAMERA_USER']}?password={config['CAMERA_PASSWORD']}",
    # ]

    # Generate camera URLs dynamically
    CAMERAS = [
        f"http://{ip}/mjpeg.cgi?user={config['CAMERA_USER']}&password={config['CAMERA_PASSWORD']}"
        for ip in config["CAMERAS"].values()
    ]

    # COCO classes initialization
    CLASSES = [
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    DETECT_CLASSES = {"person": CLASSES.index("person"), "cat": CLASSES.index("cat")}
    CONFIDENCE = 0.5  # Confidence threshold for detections

    ANALYZE_EVERY_N_SECONDS = 2  # Analyze a frame every N seconds
    CROP_MARGINS = {"top": 10, "bottom": 200, "left": 0, "right": 200}


## Iterate over the keys of DETECT_CLASSES to generate folder names and ensure they exist
#for class_name in Config.DETECT_CLASSES.keys():
#   folder_name = f"{class_name}_folder"  # Generate folder name
#   os.makedirs(folder_name, exist_ok=True)  # Ensure the folder exists


# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)

## Load the object detection model
# net = cv2.dnn.readNetFromTensorflow(Config.MODEL_PATH, Config.CONFIG_PATH)
## net = cv2.dnn.readNetFromONNX("cctv_classification/yolov8x.onnx")

# Initialize the HOG descriptor
hog = cv2.HOGDescriptor()
# Set the SVM detector for people detection
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def read_image(image_path):
    image = None
    try:
        image = cv2.imread(image_path)
    except Exception as e:
        logging.error(f"Error reading image: {e}")

    # cv2.imshow('GFG', image)
    return image


def crop_image(image, crop_margins):

    if crop_margins:
        h, w = image.shape[:2]  # Get image height and width
        cropped_image = image[  # Crop the image based on the margins
            crop_margins.get("top", 0) : h - crop_margins.get("bottom", 0),
            crop_margins.get("left", 0) : w - crop_margins.get("right", 0),
        ]
        image = cropped_image
    return image


# Function to analyze an image and return the detected object type
def analyze_image(frame):
    conf_threshold = Config.CONFIDENCE  # Minimum confidence threshold for detections
    detected_objects = []  # List to store detected objects and their bounding boxes

    # Start time before the detection
    start_time = time.time()

    # Ensure the image was loaded correctly
    if frame is None:
        print("Not a valid image")
        return None, None

    # blob = cv2.dnn.blobFromImage(
    #    frame,
    #    swapRB=True,
    #    crop=False,
    #    size=(640, 640),  # YOLOv8x model input size
    # )  # Create a blob from the image
    #
    # net.setInput(blob)  # Set the input for the neural network
    # detections = net.forward()  # Perform a forward pass of the neural network

    (H, W) = frame.shape[:2]  # Get the height and width of the image

    # https://gist.github.com/leandrobmarinho/26bd5eb9267654dbb9e37f34788486b5
    # https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c

    # for i in range(detections.shape[2]):
    #    confidence = detections[0, 0, i, 2]
    #    class_id = int(detections[0, 0, i, 1])
    #
    #    if confidence > conf_threshold:
    #        ## Iterate through DETECT_CLASSES to find a match for the detected class_id
    #        # for object_type, id in Config.DETECT_CLASSES.items():
    #        #    if class_id == id:
    #        #        # Update detected_type and highest_confidence if this detection has the highest confidence so far
    #        #        if confidence > highest_confidence:
    #        #            detected_type = object_type
    #        #            highest_confidence = confidence
    #
    #        # Get the bounding box coordinates
    #        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
    #        (startX, startY, endX, endY) = box.astype("int")
    #        # https://stackoverflow.com/questions/59409692/return-type-of-net-forward
    #
    #        # Iterate through DETECT_CLASSES to find a match for the detected class_id
    #        for object_type, id in Config.DETECT_CLASSES.items():
    #            if class_id == id:
    #                detected_objects.append(
    #                    {
    #                        "type": object_type,
    #                        "confidence": confidence,
    #                        "bbox": [startX, startY, endX, endY],
    #                    }
    #                )

    ## reduce size of image to speed up detection
    # frame = cv2.resize(frame, (640, 480))
    ## convert the image to grayscale to speed up detection
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, confidences = hog.detectMultiScale(frame, winStride=(8, 8))
    
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # End time after the detection
    end_time = time.time()

    # Calculate the time of execution
    execution_time = end_time - start_time

    for i in range(len(boxes)):
        if confidences[i] > conf_threshold:
            [startX, startY, endX, endY] = boxes[i].astype("int")
            detected_objects.append(
                {
                    "type": "person",
                    "confidence": confidences[i],
                    "bbox": [startX, startY, endX, endY],
                    "execution_time": execution_time,
                }
            )

    return detected_objects


def show_rectangles(image, detected_object):
    # Extract object type, confidence, and bounding box
    object_type = detected_object["type"]
    confidence = detected_object["confidence"]
    bbox = detected_object["bbox"]
    startX, startY, endX, endY = bbox
    # Draw bounding box and add text
    # parameters: image, start_point, end_point, color, thickness
    # color parameter: BGR format
    cv2.rectangle(image, (startX, startY), (endX, endY), COLOR, THICKNESS)

    # Add text with object type and confidence
    text = f"{object_type}: {confidence:.2f}"
    cv2.putText(
        image,
        text,
        (startX, startY - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR,
        THICKNESS,
    )

    return image


def copy_image(image_path, destination_folder, new_filename):

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    # Move the image to the determined folder
    # move(image_path, os.path.join(destination_folder, new_filename))
    copy(image_path, os.path.join(destination_folder, new_filename))


def write_image(annotated_image, destination_folder, new_filename):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    # Using cv2.imwrite() method
    # Saving the image
    try:
        cv2.imwrite(os.path.join(destination_folder, new_filename), annotated_image)
    except Exception as e:
        logging.error(f"Error writing image: {e}")
