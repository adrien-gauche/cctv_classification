from cctv_classification import *
import cv2  # For image processing
import os  # For file operations
import logging  # For logging
from shutil import move  # For moving files
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import pandas as pd


# Initialize an empty list to log detections
detections = []

# Recursively explore the folder structure and process images
for root, dirs, files in os.walk(base_folder):
    # for file in files:
    for file in tqdm(files):
        if file.endswith(".jpg"):
            date = os.path.basename(
                os.path.normpath(root)
            )  # Extract date from the folder name
            time = file.split(".")[0]  # Extract time from the file name
            image_path = os.path.join(root, file)
            image = get_image(image_path)
            image = crop_image(image, Config.CROP_MARGINS)
            # cv2.imshow('GFG', image)
            object_type, confidence = analyze_image(image)
            copy_image(image_path, date, time, object_type, confidence)
            if object_type in Config.DETECT_CLASSES.keys():
                # Convert date and time into a datetime object with the correct format
                datetime_str = date + " " + time  # Ensure time format is HH:MM:SS
                datetime = pd.to_datetime(datetime_str, format="%Y-%m-%d %H-%M-%S")

                # Log the detection in the dictionary
                detections.append(
                    {
                        "datetime": datetime,
                        "object_type": object_type,
                        "confidence": confidence,
                    }
                )
                print(pd.DataFrame(detections))

            #    print(f"Object type: {object_type}, Confidence: {confidence:.2f}", image_path)
            #    cv2.imshow('GFG', image)
            #    # pause with keyboard binding
            #    cv2.waitKey(0)
            #    # destroy the window
            #    cv2.destroyAllWindows()

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(detections)
df.set_index("datetime", inplace=True)

# Save the DataFrame to a CSV file
df.to_csv("detections.csv")
