import cv2
import numpy as np
import imutils
from sahi.slicing import slice_image
from PIL import Image
import os
import random
import shutil
import pandas as pd

def slicing(input_folder, output_directory, name_slicing, number_pictures, train_percent=60, val_percent=20, 
            slice_width=640, slice_height=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, crop="NA", crop_level=2):
    """
    Function to slice images from an input folder into smaller parts and split them into training, validation, and test sets.
    
    Parameters:
    - input_folder (str): Path to the folder containing the images.
    - output_directory (str): Path to the folder where sliced images will be saved.
    - name_slicing (str): Name of the output slicing folder.
    - number_pictures (int): Number of images to be processed.
    - train_percent (int): Percentage of images for training (default: 60%).
    - val_percent (int): Percentage of images for validation (default: 20%).
    - slice_width (int): Width of each image slice.
    - slice_height (int): Height of each image slice.
    - overlap_height_ratio (float): Overlap ratio for height between slices.
    - overlap_width_ratio (float): Overlap ratio for width between slices.
    - crop (str): Crop type ("left", "right", or "NA").
    - crop_level (int): Crop intensity (default: 2, meaning half the image).
    
    Returns:
    - list_slices (list): A list containing the sliced images.
    """
    image_list = os.listdir(input_folder)
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_list = [file for file in image_list if file.lower().endswith(tuple(image_extensions))]
    
    if len(image_list) < number_pictures:
        print("The folder does not contain enough images.")
        return
    
    random_pictures = random.sample(image_list, number_pictures)
    list_slices = []
    
    # Define output directories
    output_folder = os.path.join(output_directory, name_slicing)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    
    # If the folder already exists, delete its contents
    if os.path.exists(output_folder):
        shutil.rmtree(train_folder)
        shutil.rmtree(val_folder)
        shutil.rmtree(test_folder)
    
    # Create necessary folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Determine number of images per subset
    num_train = int(train_percent / 100 * number_pictures)
    num_val = int(val_percent / 100 * number_pictures)
    
    
    # Create image subsets
    train_images = random_pictures[:num_train]
    val_images = random_pictures[num_train:num_train + num_val]
    
    
    for image_input in random_pictures:
        image_path = os.path.join(input_folder, image_input)
        image_selected = Image.open(image_path)
        image_name, extension = os.path.splitext(image_input)
        
        # If the image is large and the object is only on the left or right, crop accordingly
        if crop == "left":
            width, height = image_selected.size
            image_selected = image_selected.crop((0, 0, width // crop_level, height))
        elif crop == "right":
            width, height = image_selected.size
            image_selected = image_selected.crop((width // crop_level, 0, width, height))
        
        # Determine output folder based on dataset type
        if image_input in train_images:
            output_subfolder = train_folder
        elif image_input in val_images:
            output_subfolder = val_folder
        else:
            output_subfolder = test_folder
        
        # Slice the image and save the slices
        sliced = slice_image(image=image_selected, slice_width=slice_width, slice_height=slice_height, 
                             overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio, 
                             output_dir=output_subfolder, verbose=True, output_file_name=f"SL_{image_name}")
        
        list_slices.append(sliced)
    
    return list_slices


def obtain_pixel_metric(info_data, contours, output_directory, reference=24.25, smoothing=False,
                        smoothing_kernel=3, smoothing_iterations=1):
    """
    Calculates the pixel-to-metric conversion for given image contours.

    Parameters:
    - info_data: DataFrame containing image information.
    - contours: List of tuples containing (mask, image_path).
    - output_directory: Directory to save the processed data.
    - reference: Real-world reference length for metric conversion (default: 24.25).
    - smoothing: Boolean, applies morphological operations to smooth the contour (default: False).
    - smoothing_kernel: Kernel size for morphological operations (default: 3).
    - smoothing_iterations: Number of times to apply smoothing operations (default: 1).

    Returns:
    - A DataFrame with pixel-to-metric conversion values merged with the original info_data.
    """
    
    pixel_metric_list = []

    for contour in contours:
        name_pic = os.path.basename(contour[1])  # Extract the image filename
        mask = contour[0]

        # Apply morphological operations if smoothing is enabled
        if smoothing:
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (smoothing_kernel, smoothing_kernel))
            mask = cv2.erode(mask, rect_kernel, iterations=smoothing_iterations)
            mask = cv2.dilate(mask, rect_kernel, iterations=smoothing_iterations)      

        # Find contours from the mask
        mask_contours_list, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the largest contour
        i = 0
        for contour_opencv in mask_contours_list:
            if len(mask_contours_list) > 1:
                area = cv2.contourArea(contour_opencv)
                if i == 0:
                    max_contour_area = contour_opencv
                    max_area = area
                elif area > max_area:
                    max_contour_area = contour_opencv
                i += 1
            else:
                max_contour_area = contour_opencv
        
        # Compute the minimum area bounding box
        box = cv2.minAreaRect(max_contour_area)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Compute the perimeter of the bounding box and derive the average diameter
        perimeter_reference = cv2.arcLength(box, True)
        average_diameter = perimeter_reference / 4  # Approximate diameter from the perimeter

        # Calculate the pixels-to-metric conversion factor
        pixelsPerMetric = average_diameter / reference
        pixel_metric_list.append([name_pic, pixelsPerMetric])

    # Create a DataFrame from the calculated metrics
    df_pix_met = pd.DataFrame(pixel_metric_list, columns=['Name_picture', 'Pixelmetric'])

    # Merge with the original info_data DataFrame
    info_data_completed = pd.merge(info_data, df_pix_met, on='Name_picture')

    # Save the updated data to a text file
    output = os.path.join(output_directory, "info_data_completed.txt")
    info_data_completed.to_csv(output, index=False, sep='\t')

    return info_data_completed


def divide_in_sets(input_folder, output_directory, number_pictures, division_name, 
                   train_percent=60, val_percent=20):
    """
    Divides a set of images into training, validation, and test sets.

    Parameters:
    - input_folder: Path to the folder containing images.
    - output_directory: Path to store the divided datasets.
    - number_pictures: Total number of images to be selected and divided.
    - division_name: Name for the output folder containing the divided sets.
    - train_percent: Percentage of images for the training set (default: 60).
    - val_percent: Percentage of images for the validation set (default: 20).
    

    Returns:
    - None (The images are copied to the respective folders).
    """

    # Get the list of images
    picture_list = os.listdir(input_folder)
    image_extensions = ['.jpg', '.jpeg', '.png']
    picture_list = [file for file in picture_list if file.lower().endswith(tuple(image_extensions))]
    
    # Check if there are enough images
    if len(picture_list) < number_pictures:
        print("The folder does not contain enough images.")
        return
    
    # Randomly select a subset of images
    random_pictures = random.sample(picture_list, number_pictures)
    
    # Create output folders (train, val, test)
    output_folder = os.path.join(output_directory, division_name)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    # Remove the existing folder if it already exists to avoid duplication
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create directories for train, validation, and test sets
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Determine the number of images per set
    num_train = int(train_percent / 100 * number_pictures)
    num_val = int(val_percent / 100 * number_pictures)
    num_test = number_pictures - num_train - num_val

    # Create subsets of images
    train_images = random_pictures[:num_train]
    val_images = random_pictures[num_train:num_train + num_val]

    # Move the images to their respective folders
    for image in random_pictures:
        image_path = os.path.join(input_folder, image)
        if image in train_images:
            shutil.copy(image_path, train_folder)  # Copy to the training folder
        elif image in val_images:
            shutil.copy(image_path, val_folder)    # Copy to the validation folder
        else:
            shutil.copy(image_path, test_folder)   # Copy to the test folder

    print(f"Images successfully divided: {num_train} in train, {num_val} in val, {num_test} in test.")


def get_min_xy(contour):
    """
    Returns the minimum x and y coordinates of a contour's bounding box.

    Parameters:
    - contour: A contour detected in an image.

    Returns:
    - (x, y): Tuple containing the x and y coordinates of the top-left corner of the bounding box.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return x, y


def ungroup_pic(input_contours, output_path, info_file, axis="X"):
    """
    Function to process and ungroup contours from images, crop the regions of interest, 
    and save them as new images with transparent backgrounds.

    Parameters:
    - input_contours (List): A list of objects containing contour data. Each object represents an image or mask with contours.
    - output_path (str): The path where the processed images will be saved.
    - info_file (DataFrame): A pandas DataFrame containing metadata related to the images (e.g., file names, sample numbers).
    - axis (str, optional): Specifies the axis ('X' or 'Y') by which to sort the contours before processing. Default is 'X'.
    
    Returns:
    - info_data_completed (DataFrame): A DataFrame that merges the original info_file with the new processed image data.
    """
    
    n = 1  # Initialize a counter for tracking the number of images being processed
    id_list = []  # List to store metadata about the processed images

    # Loop through all input contour pictures
    for pic in input_contours:
        try:
            # Print progress message indicating which picture is being processed
            print(f"Picture ungrouped {n}/{len(input_contours)}")
            
            # Get the file name without the extension
            pic_sin_ext = os.path.splitext(os.path.basename(pic.path))[0]
            
            list_contours_ordered = []  # List to store contours in order

            # Loop through each contour in the picture
            for contour in pic.masks.xy:
                array_contour = np.array(contour)  # Convert contour into a NumPy array
                array_contour = array_contour.reshape(-1, 2)  # Reshape into a 2D array
                contour_pixels = array_contour.astype(np.int32)  # Convert to integer type for OpenCV
                contour_opencv = contour_pixels.reshape((-1, 1, 2))  # Reshape to fit OpenCV's contour format
                list_contours_ordered.append(contour_opencv)  # Add contour to the ordered list

            # Sort the contours based on the X or Y axis depending on the specified 'axis'
            if axis == "X":
                list_contours_ordered = sorted(list_contours_ordered, key=lambda contour: get_min_xy(contour)[0])
            elif axis == "Y":
                list_contours_ordered = sorted(list_contours_ordered, key=lambda contour: get_min_xy(contour)[1])

            i = 1  # Initialize a counter for naming the output images

            # Loop through each contour in the ordered list
            for contour_ord in list_contours_ordered:
                image = cv2.imread(pic.path)  # Read the image corresponding to the current picture
                mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a black mask
                cv2.drawContours(mask, [contour_ord], -1, 255, -1)  # Draw the contour onto the mask

                # Apply the mask to extract the region of interest from the image
                region = cv2.bitwise_and(image, image, mask=mask)
                
                # Get the bounding box of the contour (the enclosing rectangle)
                x, y, w, h = cv2.boundingRect(contour_ord)
                
                # Crop the region based on the bounding box coordinates
                cropped_region = region[y:y+h, x:x+w]
                cropped_mask = mask[y:y+h, x:x+w]

                # Create an image with an alpha channel (transparent outside the contour)
                transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
                transparent_image[:, :, 0:3] = cropped_region  # Copy the cropped image
                transparent_image[:, :, 3] = cropped_mask   # Use the cropped mask as the alpha channel

                # Save the image as a PNG with adjusted dimensions based on the contour
                output_folder = os.path.join(output_path, "Ungrouped_pics")
                os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
                name_pic = f'{output_folder}/{pic_sin_ext}_{i}.png'
                cv2.imwrite(name_pic, transparent_image)  # Save the image

                # Add the picture information to the list
                id_list.append([os.path.basename(pic.path), i, f"{pic_sin_ext}_{i}.png"])
                i += 1  # Increment the image counter

        except:
            # In case of an error, print the message indicating the problem with the picture
            print(f"Problem with the picture {pic_sin_ext}")

        n += 1  # Increment the picture counter

    # Create a DataFrame from the list of processed images' information
    df_ungrouped = pd.DataFrame(id_list, columns=['Name_picture', 'Sample_number', 'Sample_picture'])
    
    # Merge the original information file with the new image data
    info_data_completed = pd.merge(info_file, df_ungrouped, on=['Name_picture', "Sample_number"])
    
    # Save the merged data to a text file
    output = os.path.join(output_path, "info_data_completed_ungrouped.txt")
    info_data_completed.to_csv(output, index=False, sep='\t')

    return info_data_completed  # Return the completed DataFrame


def midpoint(ptA, ptB):
    """
    Function to calculate the midpoint between two points.

    Parameters:
    - ptA (tuple): A tuple representing the coordinates (x, y) of the first point.
    - ptB (tuple): A tuple representing the coordinates (x, y) of the second point.

    Returns:
    - (tuple): A tuple representing the coordinates of the midpoint between the two points.
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def cart2pol(x, y):
    """
    Function to convert Cartesian coordinates (x, y) to polar coordinates (theta, rho).

    Parameters:
    - x (float): The x-coordinate in Cartesian space.
    - y (float): The y-coordinate in Cartesian space.

    Returns:
    - (theta, rho): A tuple where:
      - theta (float): The angle (in radians) between the point and the x-axis, measured counter-clockwise.
      - rho (float): The radial distance from the origin (0, 0) to the point.
    """
    theta = np.arctan2(y, x)  # Compute the angle using arctangent
    rho = np.hypot(x, y)  # Compute the distance using Pythagorean theorem
    return theta, rho


def pol2cart(theta, rho):
    """
    Function to convert polar coordinates (theta, rho) to Cartesian coordinates (x, y).

    Parameters:
    - theta (float): The angle (in radians) between the point and the x-axis, measured counter-clockwise.
    - rho (float): The radial distance from the origin (0, 0) to the point.

    Returns:
    - (x, y): A tuple representing the Cartesian coordinates (x, y) corresponding to the polar coordinates.
    """
    x = rho * np.cos(theta)  # Compute the x-coordinate from the polar angle and radius
    y = rho * np.sin(theta)  # Compute the y-coordinate from the polar angle and radius
    return x, y


def calculate_vertical_symmetry(binary_mask):
    """
    Function to calculate the vertical symmetry of a binary image.

    This function divides the binary image into two halves, compares the left half with the mirrored right half,
    and calculates a symmetry score based on the number of matching pixels.

    Parameters:
    - binary_mask (numpy.ndarray): A binary image (values of 0 and 255) where symmetry is calculated.
      The image should have pixel values 0 (black) and 255 (white), with the shape of the image being 
      (height, width).

    Returns:
    - float: A value between 0 and 1 indicating the vertical symmetry of the image.
      A score of 1 means the image is perfectly symmetrical, while a score of 0 means no symmetry.
    """
    
    # Get the dimensions of the image (height and width)
    height, width = binary_mask.shape
    
    # Split the image into the left half (up to the middle column)
    left_half = binary_mask[:, :width // 2]
    
    # If the width is odd, the right half will be adjusted by removing the center pixel
    if width % 2 != 0:
        right_half = binary_mask[:, width // 2 + 1:]
    else:
        right_half = binary_mask[:, width // 2:]

    # Flip the right half horizontally
    flipped_right_half = cv2.flip(right_half, 1)

    # Calculate the absolute difference between the left half and the flipped right half
    difference = cv2.absdiff(left_half, flipped_right_half)

    # Count the number of non-zero pixels (pixels that differ between the two halves)
    differing_pixels = cv2.countNonZero(difference)
    
    # Get the total number of pixels in the left half (or right half)
    total_pixels = left_half.size

    # Calculate the symmetry score: closer to 1 means more symmetry
    symmetry = 1 - (differing_pixels / total_pixels)
    
    return symmetry


def calculate_horizontal_symmetry(binary_mask):
    """
    Function to calculate the horizontal symmetry of a binary image.

    This function divides the binary image into two halves, compares the top half with the mirrored bottom half,
    and calculates a symmetry score based on the number of matching pixels.

    Parameters:
    - binary_mask (numpy.ndarray): A binary image (values of 0 and 255) where symmetry is calculated.
      The image should have pixel values 0 (black) and 255 (white), with the shape of the image being 
      (height, width).

    Returns:
    - float: A value between 0 and 1 indicating the horizontal symmetry of the image.
      A score of 1 means the image is perfectly symmetrical, while a score of 0 means no symmetry.
    """
    
    # Get the dimensions of the image (height and width)
    height, width = binary_mask.shape
    
    # Split the image into the top half (up to the middle row)
    top_half = binary_mask[:height // 2, :]
    
    # If the height is odd, the bottom half will be adjusted by removing the center row
    if height % 2 != 0:
        bottom_half = binary_mask[height // 2 + 1:, :]
    else:
        bottom_half = binary_mask[height // 2:, :]

    # Flip the bottom half vertically
    flipped_bottom_half = cv2.flip(bottom_half, 0)

    # Calculate the absolute difference between the top half and the flipped bottom half
    difference = cv2.absdiff(top_half, flipped_bottom_half)

    # Count the number of non-zero pixels (pixels that differ between the two halves)
    differing_pixels = cv2.countNonZero(difference)
    
    # Get the total number of pixels in the top half (or bottom half)
    total_pixels = top_half.size

    # Calculate the symmetry score: closer to 1 means more symmetry
    symmetry = 1 - (differing_pixels / total_pixels)
    
    return symmetry

 
def smoothing_masks(mask, smoothing_kernel, smoothing_iterations):
    """
    Applies morphological operations to smooth the mask by performing erosion followed by dilation.

    Parameters:
        mask (numpy.ndarray): The binary mask to be smoothed.
        smoothing_kernel (int): Size of the square kernel used for morphological operations.
        smoothing_iterations (int): Number of iterations for erosion and dilation.

    Returns:
        mask (numpy.ndarray): The smoothed binary mask after morphological operations.
        rect_kernel (numpy.ndarray): The rectangular structuring element used for smoothing.
    """
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (smoothing_kernel, smoothing_kernel))
    mask = cv2.erode(mask, rect_kernel, iterations=smoothing_iterations)
    mask = cv2.dilate(mask, rect_kernel, iterations=smoothing_iterations)
    return mask, rect_kernel


def watershed(mask, rect_kernel, iterations, kernel_watershed, threshold_watershed):
    """
    Applies the Watershed algorithm to perform image segmentation.

    Parameters:
        mask (numpy.ndarray): The binary mask representing the segmented image.
        rect_kernel (numpy.ndarray): The structuring element used for dilation to define sure background.
        iterations (int): Number of iterations for dilation to determine sure background.
        kernel_watershed (int): Kernel size for distance transform computation.
        threshold_watershed (float): Threshold ratio (relative to max value) for foreground determination.

    Returns:
        mask (numpy.ndarray): The segmented binary mask after applying the Watershed algorithm.
    """
    # Determine the sure background
    sure_bg = cv2.dilate(mask, rect_kernel, iterations=iterations)

    # Obtain sure foreground
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, kernel_watershed)
    _, sure_fg = cv2.threshold(dist_transform, threshold_watershed * dist_transform.max(), 255, 0)

    # Find unknown areas (neither background nor foreground)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label connected components
    _, markers = cv2.connectedComponents(sure_fg)

    # Increase markers to ensure the background is labeled as 1 instead of 0
    markers = markers + 1

    # Mark unknown areas as 0
    markers[unknown == 255] = 0

    # Convert grayscale mask to color for Watershed processing
    img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # Mark the Watershed boundaries (borders) as 0
    mask[markers == -1] = 0

    # Convert the segmented result into a binary mask
    mask = np.uint8(markers > 1) * 255

    return mask