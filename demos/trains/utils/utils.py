import cv2
import random
import os
import uuid
import numpy as np

def degrad_sign(img, ratio, alpha_channel):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert image to HSV color space
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1] * ratio # scale pixel values up for channel 1
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2] * ratio # scale pixel values up for channel 2
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    random_blur = random.randint(1,3)
    img = cv2.blur(img,(random_blur,random_blur))
    #img[:, :, 3] = alpha_channel
    return img

# Function to insert image onto frame at random coordinates and scale if necessary
def insert_image(frame, image, x_range, y_range, traffic_size):
    img_height, img_width, _ = image.shape
    max_width = min(x_range, img_width)
    max_height = min(y_range, img_height)

    # Randomly choose a scale factor
    scale_factor = random.uniform(0.1, 0.3)  # You can adjust this range as needed
    
    # Scale the image
    scaled_width = int(max_width * scale_factor)
    scaled_height = int(max_height * scale_factor)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))

    # Extract alpha channel from the image
    alpha_channel = scaled_image[:, :, 3] / 255.0

    # Change brightness
    ratio = random.uniform(0.7, 1.1)
    scaled_image = degrad_sign(scaled_image, ratio, alpha_channel)
    #scaled_image = cv2.convertScaleAbs(scaled_image, alpha=1, beta=-50)

    # Randomly choose coordinates within the frame
    x = random.randint(0, x_range - scaled_width)
    y = random.randint(0, y_range - scaled_height)

    # Blend the scaled image onto the frame using alpha channel
    for c in range(0, 3):
        frame[y:y+scaled_height, x:x+scaled_width, c] = \
            alpha_channel * scaled_image[:, :, c] + \
            (1 - alpha_channel) * frame[y:y+scaled_height, x:x+scaled_width, c]

    # Draw bounding box around the inserted image
    # cv2.rectangle(frame, (x, y), (x + scaled_width, y + scaled_height), (0, 255, 0), 2)

    # Draw a square on top of the inserted image
    rect_w = int(traffic_size[0] * scale_factor)
    rect_h = int(traffic_size[1] * scale_factor)
    rect_x = x
    rect_y = y
    # cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)

    return frame, (rect_x, rect_y, rect_w, rect_h)

def to_yolo_format(image_width, image_height, box):
    """
    Convert bounding box coordinates to YOLO format.
    
    Args:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        box (tuple): Bounding box coordinates in format (x, y, width, height).
    
    Returns:
        tuple: Normalized YOLO format coordinates (x_center, y_center, width, height).
    """
    x, y, width, height = box

    # Calculate center coordinates
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height

    # Normalize width and height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height

def extract_random_frames(video_path, num_frames):
    """
    Extract randomly selected frames from a video.

    Args:
        video_path (str): Path to the input video file.
        num_frames (int): Number of frames to extract.

    Returns:
        list: List of randomly selected frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Discard start and end
    start = int(total_frames * 0.1)
    stop = int(total_frames * 0.9)

    # Ensure num_frames is not greater than the total number of frames
    num_frames = min(stop - start + 1, num_frames)

    # Randomly select frames
    selected_frames = random.sample(range(start, stop), num_frames)
    selected_frames.sort()

    # Read selected frames
    for frame_idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def add_to_yolo_dataset(frame, split, label_id, label_name, yolo_format, dataset_path):
    x, y, w, h = yolo_format
    uid = str(uuid.uuid4())
    tar_label = dataset_path + "labels/" + split + "/"
    tar_images = dataset_path + "images/" + split + "/"
    img_name = label_name + "_" + uid + ".jpg"
    label_name = label_name + "_" + uid + ".txt"
    cv2.imwrite(tar_images + img_name, frame)
    label_line = f"{label_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
    with open(tar_label + label_name, 'w') as f:
        f.write(label_line)

def determine_split(i, nb_data):
    if nb_data < 5:
        split = "train"
    else:
        split = "train" if i / nb_data < 0.8 else "val"
    return split