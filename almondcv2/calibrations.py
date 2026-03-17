from plantcv import plantcv as pcv
import os
import numpy as np
import cv2 as cv
import pathlib


def calibrate_color(input_picture="", input_folder="", output_path="", approach="color", radius_parameter=10, standard_matrix=False, force_standard_matrix=False):
    """
    Calibrates images according to the color correction approach.
    
    Parameters:
    - input_picture (str): Path to the input image (for "combined" approach).
    - input_folder (str): Folder with images to calibrate (for "color" approach).
    - output_path (str): Path where calibrated images will be saved.
    - approach (str): Calibration approach ("color" to process all images in the folder, "combined" for a single image).
    - radius_parameter (int): Radius parameter to detect the color card.
    - standard_matrix (bool): If `True`, it uses a standard matrix for color correction. Default is `False`.
    - force_standard_matrix (bool): If `True`, forces the use of the standard matrix. Default is `False`.
    
    Returns:
    - img_cc: The corrected image.
    """
    
    if approach == "color":
        errors = []
        for image_input in os.listdir(input_folder):
            if image_input.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    image_path = os.path.join(input_folder, image_input)
                    source_cv, _, _ = pcv.readimage(filename=image_path)

                    # Detect the color card
                    card_mask = pcv.transform.detect_color_card(rgb_img=source_cv, radius=radius_parameter)
                    headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=source_cv, mask=card_mask)
                    std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                    img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                                  target_matrix=std_color_matrix)
                    pcv.print_image(img=img_cc, filename=os.path.join(output_path, f"CL_{image_input}"))

                    # If force_standard_matrix is True, use the standard matrix
                    if force_standard_matrix:
                        print("Using standard matrix")
                        standard_matrix_pic, _, _ = pcv.readimage(filename=standard_matrix)
                        card_mask = pcv.transform.detect_color_card(rgb_img=standard_matrix_pic, radius=radius_parameter)
                        headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=standard_matrix_pic, mask=card_mask)
                        std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                        img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                                        target_matrix=std_color_matrix)
                        pcv.print_image(img=img_cc, filename=os.path.join(output_path, f"CL_{image_input}"))

                except Exception as e:
                    print(f"Some problem with picture {os.path.join(input_folder, f'{image_input}')}")
                    print(e)
                    errors.append(image_input)

                    # If standard_matrix is True, use it
                    if standard_matrix:
                        print("Using standard matrix")
                        standard_matrix_pic, _, _ = pcv.readimage(filename=standard_matrix)
                        card_mask = pcv.transform.detect_color_card(rgb_img=standard_matrix_pic, radius=radius_parameter)
                        headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=standard_matrix_pic, mask=card_mask)
                        std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                        img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                                        target_matrix=std_color_matrix)
                        pcv.print_image(img=img_cc, filename=os.path.join(output_path, f"CL_{image_input}"))

        # Save errors to a file
        with open(os.path.join(output_path, "errors_in_calibrations.txt"), "w") as file:
            for item in errors:
                file.write(f"{item}\n")
    
    elif approach == "combined":
        try:
            source_cv, _, _ = pcv.readimage(filename=input_picture)

            if not standard_matrix:
                # Detect the color card and perform correction without the standard matrix
                card_mask = pcv.transform.detect_color_card(rgb_img=source_cv, radius=radius_parameter)
                headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=source_cv, mask=card_mask)
                std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                              target_matrix=std_color_matrix)
            
            if standard_matrix:
                # If a standard matrix is specified, use it
                standard_matrix_pic, _, _ = pcv.readimage(filename=standard_matrix)
                card_mask = pcv.transform.detect_color_card(rgb_img=standard_matrix_pic, radius=radius_parameter)
                headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=standard_matrix_pic, mask=card_mask)
                std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                              target_matrix=std_color_matrix)

            return img_cc, input_picture
        except Exception as e:
            print(f"Some problem with picture {os.path.join(input_folder, f'{input_picture}')}")
            print(e)


def build_calibration(chessboardSize, frameSize, dir_path, image_format, size_of_chessboard_squares_mm, scale_percent=20):
    """
    This function performs camera calibration using a set of images of a chessboard pattern.
    It finds the chessboard corners, computes the camera matrix and distortion coefficients, and saves the calibration results.

    Parameters:
    - chessboardSize (tuple): The number of internal corners in the chessboard pattern (rows, columns).
    - frameSize (tuple): The size of the images (width, height).
    - dir_path (str): Path to the folder containing the chessboard images for calibration.
    - image_format (str): Image format to search for (e.g., ".jpg", ".png").
    - size_of_chessboard_squares_mm (float): The size of each square on the chessboard in millimeters.
    - scale_percent (int, optional): Percentage to scale the image for display. Default is 20. 

    Returns:
    - None: Saves the calibration matrix and distortion coefficients to a file.
    """

    # Termination criteria for corner subpixel refinement (maximum iterations and epsilon tolerance)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (3D points in real-world space), like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # Create a grid of points based on the chessboard size
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    objp = objp * size_of_chessboard_squares_mm  # Scale by the size of each square

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane (corners of the chessboard)

    # Loop through all images in the specified directory
    images = pathlib.Path(dir_path).glob(f'*{image_format}')

    for image in images:
        img = cv.imread(str(image))  # Read the image
        pic_width = int(img.shape[1])  # Image width
        pic_height = int(img.shape[0])  # Image height
        dim_pic = (pic_width, pic_height)  # Image dimensions
        img = cv.resize(img, dim_pic, interpolation=cv.INTER_AREA)  # Resize image to the specified dimensions
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert the image to grayscale

        # Find the chessboard corners in the image
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If corners are found, refine them and add them to the object and image points
        if ret == True:
            objpoints.append(objp)  # Add object points
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # Refine corner locations
            imgpoints.append(corners2)  # Add refined image points

            # Draw and display the corners on the image
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

            # Resize the image for display
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)  # Resize the image
            try:
                cv.imshow('Resized Image', resized_img)  # Show the resized image
                cv.waitKey(1000)  # Display the image for 1 second
                cv.destroyAllWindows()  # Close the image window
            except:
                from google.colab.patches import cv2_imshow
                cv2_imshow(resized_img)

    # Perform camera calibration using the collected object points and image points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # Save the camera matrix (mtx) and distortion coefficients (dist) to a compressed file
    np.savez_compressed(f'{dir_path}/calibration_mtx.npz', mtx=mtx, dist=dist)


def calibrate_distortion(mtx_input, output_path, input_folder=None, input_picture=None, approach="distortion", scale_percent=20):
    """
    Calibrates the distortion of images using a pre-calibrated camera matrix and distortion coefficients.

    Parameters:
    - mtx_input (str): Path to the file containing the camera matrix and distortion coefficients.
    - output_path (str): Path to save the undistorted images.
    - input_folder (str, optional): Folder containing the images to calibrate (for "distortion" approach).
    - input_picture (tuple, optional): A tuple (image, image_filename) for the "combined" approach.
    - approach (str, optional): The approach to use. "distortion" to calibrate images from a folder, "combined" to calibrate a single image. Default is "distortion".
    - scale_percent (int, optional): Percentage to scale the images for display. Default is 20.

    Returns:
    - None: Saves the undistorted images to the specified output path.
    """

    # Load the camera matrix and distortion coefficients
    data = np.load(mtx_input)
    mtx = data['mtx']
    dist = data['dist']

    # Distortion approach - process images from a folder
    if approach == "distortion":
        for image_input in os.listdir(input_folder):
            if image_input.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(input_folder, image_input)
                img, _, _ = pcv.readimage(filename=image_path)
                h, w = img.shape[:2]

                # Get the optimal new camera matrix
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # Undistort the image
                mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
                dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

                # Crop the image based on the ROI
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]

                # Resize the image for display if scale_percent is provided
                width = int(dst.shape[1] * scale_percent / 100)
                height = int(dst.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_img = cv.resize(dst, dim, interpolation=cv.INTER_AREA)
                try:
                    cv.imshow('Resized Image', resized_img)
                    cv.waitKey(1000)  # Show the image for 1 second
                    cv.destroyAllWindows()  # Close the window
                except:
                    from google.colab.patches import cv2_imshow
                    cv2_imshow(resized_img)
            

                # Save the undistorted image
                cv.imwrite(os.path.join(output_path, f"CL_{image_input}"), dst)

    # Combined approach - process a single image
    elif approach == "combined":
        h, w = input_picture[0].shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort the image
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv.remap(input_picture[0], mapx, mapy, cv.INTER_LINEAR)

        # Crop the image based on the ROI
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # Resize the image for display if scale_percent is provided
        width = int(dst.shape[1] * scale_percent / 100)
        height = int(dst.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv.resize(dst, dim, interpolation=cv.INTER_AREA)
        try:
            cv.imshow('Resized Image', resized_img)
            cv.waitKey(1000)  # Show the image for 1 second
            cv.destroyAllWindows()  # Close the window
        except:
            from google.colab.patches import cv2_imshow
            cv2_imshow(resized_img)

        # Save the undistorted image
        cv.imwrite(os.path.join(output_path, f"CL_{os.path.basename(input_picture[1])}"), dst)


def calibrate_color_and_distortion(raw_folder, mtx_input_path, output_calibrated, radius_param=10, standard_matrix=False, scale_percent=20):
    """
    Calibrates both the color and distortion of images from a folder.

    Parameters:
    - raw_folder (str): Folder containing the raw images to be calibrated.
    - mtx_input_path (str): Path to the file containing the camera matrix and distortion coefficients.
    - output_calibrated (str): Folder to save the calibrated images.
    - radius_param (int, optional): Parameter for color calibration. Default is 10.
    - standard_matrix (str, optional): Path to a standard color matrix. Default is "No".
    - scale_percent (int, optional): Percentage to scale the images for display. Default is 20.

    Returns:
    - None: Saves the color and distortion corrected images to the output folder.
    """

    errors = []
    for image_input in os.listdir(raw_folder):
        if image_input.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(raw_folder, image_input)
            try:
                # First, calibrate the color
                color_calibrated = calibrate_color(input_picture=image_path, approach="combined", radius_parameter=radius_param, standard_matrix=standard_matrix)

                # Then, calibrate the distortion
                calibrate_distortion(input_picture=color_calibrated, mtx_input=mtx_input_path, output_path=output_calibrated, approach="combined", scale_percent=scale_percent)
            except Exception as e:
                print(f"Some problem with picture {image_input}")
                print(e)
                errors.append(image_input)

    # Save errors to a text file
    with open(os.path.join(output_calibrated, "errors_in_calibrations.txt"), "w") as archivo:
        for item in errors:
            archivo.write(f"{item}\n")