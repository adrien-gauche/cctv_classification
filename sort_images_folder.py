#!/usr/bin/env python3

# call in terminal: python3 sort_images_folder.py -f Terrasse
# or: ./sort_images_folder.py --folder Jardin

from cctv_classification import *
import argparse
import sys


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process CCTV images for object detection."
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="Terrasse",
        help="Base folder containing the date-named folders with images",
    )
    args = parser.parse_args()

    # Base folder containing the date-named folders with images
    base_folder = args.folder

    # Check if the directory exists
    if not os.path.exists(base_folder):
        print(f"Error: The directory '{base_folder}' does not exist.")
        sys.exit(1)  # Exit the script with an error status

    # Initialize an empty list to log detections
    detections = []

    # Recursively explore the folder structure and process images
    for root, dirs, files in os.walk(base_folder):
        # os.walk() generates the file names in a directory tree by walking either top-down or bottom-up

        # for file in files:
        for file in tqdm(files):
            if file.endswith(".jpg"):

                date = os.path.basename(
                    os.path.normpath(root)
                )  # Extract date from the folder name

                time = file.split(".")[0]  # Extract time from the file name

                image_path = os.path.join(root, file)  # Full path to the image

                image = get_image(image_path)
                image = crop_image(image, Config.CROP_MARGINS)

                # DEBUG
                # cv2.imshow('GFG', image) # Display the image DEBUG
                # cv2.waitKey(0)  # Wait for a key press to continue
                # cv2.destroyAllWindows()  # Close the image window

                detected_objects = analyze_image(
                    image
                )  # This function is assumed to return a list of detections with bounding boxes

                if detected_objects:
                    for obj in detected_objects:
                        object_type = obj["type"]
                        confidence = obj["confidence"]
                        bbox = obj["bbox"]

                        image = show_rectangles(image, obj)

                        # copy_image(image_path, date, time, object_type, confidence)
                        write_image(image, date, time, object_type, confidence)

                        # Convert date and time into a datetime object with the correct format
                        datetime_str = (
                            date + " " + time
                        )  # Ensure time format is HH:MM:SS
                        datetime = pd.to_datetime(
                            datetime_str, format="%Y-%m-%d %H-%M-%S"
                        )

                        # Log the detection
                        detections.append(
                            {
                                "datetime": datetime,
                                "object_type": object_type,
                                "confidence": confidence,
                                "bbox": bbox,
                            }
                        )
                    print(pd.DataFrame(detections))

                    ## Display the image with bounding boxes
                    # cv2.imshow('Detected Objects', image)
                    # cv2.waitKey(0)  # Wait for a key press to continue
                    # cv2.destroyAllWindows()  # Close the image window

    # Convert the list of detections into a DataFrame
    df = pd.DataFrame(detections)
    df.set_index("datetime", inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv("detections.csv", index=True)

    # Optionally print DataFrame for visual confirmation
    print(df)


if __name__ == "__main__":
    main()
