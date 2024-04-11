import cv2  # For image processing
import os  # For file operations
import logging  # For logging
from shutil import move, copy  # For moving files
import matplotlib.pyplot as plt

import json  # For loading the configuration file


def load_config(config_path="config.json"):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


config = load_config()  # Load the configuration file

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Base folder containing the date-named folders with images
base_folder = "Terrasse"


# Centralized configuration class
class Config:
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # Paths to the TensorFlow object detection model files
    # https://github.com/ChiekoN/OpenCV_SSD_MobileNet/tree/master/model
    MODEL_PATH = "cctv_classification/frozen_inference_graph.pb"
    CONFIG_PATH = "cctv_classification/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

    ## use configuration from config.json
    #CAMERAS = [
    #    f"http://{config['CAMERA1_IP']}/mjpeg.cgi?user={config['CAMERA_USER']}?password={config['CAMERA_PASSWORD']}",
    #    f"http://{config['CAMERA2_IP']}/mjpeg.cgi?user={config['CAMERA_USER']}?password={config['CAMERA_PASSWORD']}",
    #]

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
    DETECT_CLASSES = {
        "person": CLASSES.index("person"),
        # "cat": CLASSES.index("cat")
    }
    CONFIDENCE = 0.2  # Confidence threshold for detections

    ANALYZE_EVERY_N_SECONDS = 2  # Analyze a frame every N seconds
    CROP_MARGINS = {"top": 0, "bottom": 0, "left": 0, "right": 150}


# Iterate over the keys of DETECT_CLASSES to generate folder names and ensure they exist
for class_name in Config.DETECT_CLASSES.keys():
    folder_name = f"{class_name}_folder"  # Generate folder name
    os.makedirs(folder_name, exist_ok=True)  # Ensure the folder exists


# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)

# Load the object detection model
net = cv2.dnn.readNetFromTensorflow(Config.MODEL_PATH, Config.CONFIG_PATH)


def get_image(image_path):
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
def analyze_image(image):
    confidence_threshold = Config.CONFIDENCE
    detected_type = "none"
    highest_confidence = 0

    # Ensure the image was loaded correctly
    if image is None:
        print("Not a valid image")
        return None, None

    blob = cv2.dnn.blobFromImage(
        image, swapRB=True, crop=False
    )  # Create a blob from the image
    net.setInput(blob)  # Set the input for the neural network
    detections = net.forward()  # Perform a forward pass of the neural network

    for i in range(detections.shape[2]):
        class_id = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Iterate through DETECT_CLASSES to find a match for the detected class_id
            for object_type, id in Config.DETECT_CLASSES.items():
                if class_id == id:
                    # Update detected_type and highest_confidence if this detection has the highest confidence so far
                    if confidence > highest_confidence:
                        detected_type = object_type
                        highest_confidence = confidence

    return detected_type, highest_confidence


def copy_image(image_path, date, time, object_type, confidence=None):
    # Check if the object type is one of the detected classes
    if object_type in Config.DETECT_CLASSES.keys():
        # new_filename = f"{date}_{time}_{confidence:.2f}.jpg"
        ## Determine the destination folder based on the object type, following the naming convention
        # destination_folder = f"{object_type}_folder"

        new_filename = f"{date}_{confidence:.2f}.jpg"
        # Adjust the destination folder to include both object type and date
        destination_folder = os.path.join(f"{object_type}_folder", date)

        # Ensure the destination folder exists
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)

        # Move the image to the determined folder
        # move(image_path, os.path.join(destination_folder, new_filename))
        copy(image_path, os.path.join(destination_folder, new_filename))
