"""
functions_processing.py

Module for processing hyperspectral images and performing segmentation.

Main functionalities:
- Extract image information and metadata from files.
- Generate pseudo-RGB images from selected spectral bands.
- Perform segmentation using model-based or watershed methods.
- Support batch processing, mask customization, and saving results.
"""

import os
import json
import cv2
import numpy as np
import sys
import traceback
sys.path.append(os.path.abspath("almondcv2")) #Add subroute
from model_class import ModelSegmentation
from aux_functions import smoothing_masks, watershed


def process_picture_info(row, session, path_session_1, subdir="HYP/RAW/", metadata=False):
    """
    Extracts information and paths for a hyperspectral image, optionally loading metadata.

    Parameters:
    - row: pd.Series
        Row of a DataFrame containing information about the sample/image.
    - session: str
        Identifier of the session.
    - path_session_1: str
        Base path to the session folder.
    - subdir: str, default "HYP/RAW/"
        Subdirectory containing the hyperspectral images.
    - metadata: bool, default False
        If True, load additional metadata from a JSON file.

    Returns:
    - If metadata=False: hyp_pth, array_shape, sample_name, pic
    - If metadata=True: hyp_pth, array_shape, sample_name, pic, nonzero_indices, original_shape, n_element
    """

    # Print basic information
    print(f"Picture number: {row.name}", f"Session: {session}")

    # Build full path to the hyperspectral image
    pic = row["Name_picture_HYP"]
    hyp_pth = os.path.join(path_session_1, subdir, pic + ".lz4")

    # Convert "Array_shape" string to a list of integers
    array_shape = [int(num) for num in row["Array_shape"].strip("()").split(", ")]

    # Get sample/individual name
    sample_name = row["Individual_name"]

    if metadata:
        # Load metadata from corresponding JSON file
        json_path = os.path.join(path_session_1, subdir, pic + ".json")
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            array_shape = metadata["shape"]
            nonzero_indices = metadata["nonzero_indices"]
            original_shape = metadata["original_shape"]
            n_element = metadata["n_element"]

        return hyp_pth, array_shape, sample_name, pic, nonzero_indices, original_shape, n_element
    else:
        return hyp_pth, array_shape, sample_name, pic

def segment_image(segment_type, pic, hyp_pic_calibrated, band_index, masks_path, mask_color, 
                  transparency_level, segmented_pseudorgb_directory, batch=True, watershed_for_indv=False, **kwargs):
    

    """
    Segments a hyperspectral image using either manual masks or AI-based models, and creates pseudorgb overlays.

    Parameters:
    - segment_type: str
        Type of segmentation to perform: 'manual_masks', 'model_ai', or 'model_ai_sahi'.
    - pic: str
        Image name.
    - hyp_pic_calibrated: np.ndarray
        Calibrated hyperspectral image (3D array: height x width x bands).
    - band_index: int
        Index of the spectral band to use for creating a pseudo-RGB image.
    - masks_path: str
        Directory containing manual masks (for 'manual_masks' mode).
    - mask_color: str
        Color to overlay the mask: 'red', 'green', or 'blue'.
    - transparency_level: float
        Alpha blending for mask overlay.
    - segmented_pseudorgb_directory: str
        Directory to save the segmented pseudorgb images.
    - batch: bool, default True
        If True, saves images in batch mode; otherwise, processes individually elements with contour labeling.
    - watershed_for_indv: bool, default False
        Apply watershed refinement to individual masks (only for non-batch mode).
    - **kwargs: dict
        Additional arguments for model-based segmentation, smoothing, and watershed parameters.

    Returns:
    - For batch mode: the masked pseudorgb image or binary mask.
    - For non-batch mode: list of contours detected in the image.
    
    Notes:
    - Creates pseudorgb images from a single selected spectral band.
    - Supports smoothing and watershed refinement for better individual object separation.
    - Works with both pre-existing manual masks and AI-predicted masks.
    - Saves segmented images to the specified directory.
    """
    # Create pseudorgb
    selected_band = hyp_pic_calibrated[:, :, band_index] 
    false_rgb = np.repeat(selected_band[:, :, np.newaxis], 3, axis=-1) 
    false_rgb = cv2.normalize(false_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Mask according to color selected
    color_mask = np.zeros_like(false_rgb)
    if mask_color == 'red':
        color_mask[:, :, 2] = 255  # Red
    elif mask_color == 'green':
        color_mask[:, :, 1] = 255  # Green
    elif mask_color == 'blue':
        color_mask[:, :, 0] = 255  # Blue


    # Provide masks performed manually in png
    if segment_type == "manual_masks":
        # Load mask
        mask = cv2.imread(f'{masks_path}/{pic}.png')
        if mask is None:
            print(f"Error: Could not load the mask for {pic}")
            return

        # To gray
        mask_gray_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Segment
        mask_applied = cv2.bitwise_and(color_mask, color_mask, mask=mask_gray_2d)
        pseudorgb_masked = cv2.addWeighted(false_rgb, 1 - transparency_level, mask_applied, transparency_level, 0)



        if batch == False: # Find element contours 
            watershed_for_indv = kwargs.get('watershed_for_indv', watershed_for_indv)
            if watershed_for_indv == True:
                smooting_iterations = kwargs.get('smooting_iterations', 2)
                smoothing_kernel = kwargs.get('smoothing_kernel', 5)
                kernel_watershed = kwargs.get('kernel_watershed', 5)
                threshold_watershed = kwargs.get('threshold_watershed', 0.6)
                watershed_iterations = kwargs.get('watershed_iterations', 3)
                mask_gray_2d, rect_kernel=smoothing_masks(mask=mask_gray_2d,smoothing_iterations=smooting_iterations, smoothing_kernel=smoothing_kernel)
                mask_gray_2d=watershed(mask=mask_gray_2d, rect_kernel=rect_kernel, iterations=watershed_iterations,
                kernel_watershed=kernel_watershed, threshold_watershed=threshold_watershed)


            contours, _ = cv2.findContours(mask_gray_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

            # Draw contours and put numbers
            for i, contour in enumerate(contours):
                cv2.drawContours(pseudorgb_masked, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                centro_x = x + w // 2
                centro_y = y + h // 2
                cv2.putText(pseudorgb_masked, str(i), (centro_x, centro_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Save
            cv2.imwrite(os.path.join(segmented_pseudorgb_directory, f"segm_{pic}.png"), pseudorgb_masked)

            return contours
        else:
            # Save
            cv2.imwrite(os.path.join(segmented_pseudorgb_directory, f"segm_{pic}.png"), pseudorgb_masked)

            return mask_gray_2d

    # Employ segmentation using ai according to AlmondCV2
    elif segment_type == "model_ai" or segment_type == "model_ai_sahi":
        print("Processing with model_ai (batch mode)...")
        try:

            # Pseudo RGB with one band. These lines could be improved selecting more bands than one. Moreover, be careful with band cutting because index changes.
            selected_band = hyp_pic_calibrated[:, :, band_index] 
            false_rgb = np.repeat(selected_band[:, :, np.newaxis], 3, axis=-1)  
            false_rgb = cv2.normalize(false_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            working_directory = kwargs.get('working_directory', None)  
            img_size = kwargs.get('img_size', 640)
            model_path = kwargs.get('model_path', None)
            overlap = kwargs.get('overlap', 0.2)
            retina_mask = kwargs.get('retina_mask', True)


            # Without SAHI
            if segment_type == "model_ai":
                conf=kwargs.get('conf', 0.5)
                model=ModelSegmentation(working_directory=working_directory)
                masks=model.slice_predict_reconstruct(imgsz=img_size, model_path=model_path,image_array=false_rgb,
                                                slice_height=img_size, slice_width=img_size,overlap_height_ratio=overlap,
                                                overlap_width_ratio=overlap, retina_mask=retina_mask, conf=conf)
                
                # Use mask
                mask_applied = cv2.bitwise_and(color_mask, color_mask, mask=masks[0][0])
                pseudorgb_masked = cv2.addWeighted(false_rgb, 1 - transparency_level, mask_applied, transparency_level, 0)

                # This part is so repetitive, could be improved
                if batch == False:

                    watershed_for_indv = kwargs.get('watershed_for_indv', watershed_for_indv)
                    if watershed_for_indv == True:
                        smooting_iterations = kwargs.get('smooting_iterations', 2)
                        smoothing_kernel = kwargs.get('smoothing_kernel', 5)
                        kernel_watershed = kwargs.get('kernel_watershed', 5)
                        threshold_watershed = kwargs.get('threshold_watershed', 0.6)
                        watershed_iterations = kwargs.get('watershed_iterations', 3)

                        mask_gray_2d, rect_kernel=smoothing_masks(mask=masks[0][0],smoothing_iterations=smooting_iterations, smoothing_kernel=smoothing_kernel)
                        mask_gray_2d=watershed(mask=mask_gray_2d, rect_kernel=rect_kernel, iterations=watershed_iterations,
                        kernel_watershed=kernel_watershed, threshold_watershed=threshold_watershed)
                    else:
                        mask_gray_2d=masks[0][0]

                    contours, _ = cv2.findContours(mask_gray_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

                    for i, contour in enumerate(contours):
                        cv2.drawContours(pseudorgb_masked, [contour], -1, (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(contour)
                        centro_x = x + w // 2
                        centro_y = y + h // 2
                        cv2.putText(pseudorgb_masked, str(i), (centro_x, centro_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imwrite(os.path.join(segmented_pseudorgb_directory, f"segm_{pic}.png"), pseudorgb_masked)

                    return contours

                else:
                    cv2.imwrite(os.path.join(segmented_pseudorgb_directory, f"segm_{pic}.png"), pseudorgb_masked)
                    return masks[0][0]
            
            # Segment using AI using SAHI approach AlmondCV2
            elif segment_type == "model_ai_sahi":

                check_result = kwargs.get('check_result', False)
                postprocess_match_threshold = kwargs.get('postprocess_match_threshold', 0.2)
                postprocess_match_metric = kwargs.get('postprocess_match_metric', "IOS")
                postprocess_type = kwargs.get('postprocess_type', "GREEDYNMM")
                postprocess_match_metric = kwargs.get('postprocess_match_metric', "IOS")
                confidence_treshold = kwargs.get('confidence_treshold', 0.95)


                model=ModelSegmentation(working_directory=working_directory)
                masks=model.predict_model_sahi(model_path=model_path, check_result=check_result, image_array=false_rgb,
                                                            retina_masks=retina_mask,
                                                            postprocess_match_threshold=postprocess_match_threshold, overlap_height_ratio=overlap,
                                                                overlap_width_ratio=overlap, postprocess_match_metric=postprocess_match_metric, 
                                                                postprocess_type=postprocess_type, slice_height=img_size, slice_width=img_size,
                                                                confidence_treshold=confidence_treshold,
                                                                imgsz=img_size)
                
                mask_contours_list=masks[0][0].object_prediction_list
                sahi_contours_list=[]
                
                for contour in mask_contours_list:
                    contour=contour.mask.segmentation[0]
                    array_contour=np.array(contour)
                    array_contour=array_contour.reshape(-1,2)
                    contour_pixels = array_contour.astype(np.int32)
                    contour = contour_pixels.reshape((-1, 1, 2))
                    sahi_contours_list.append(contour)

                contours = sorted(sahi_contours_list, key=lambda c: cv2.boundingRect(c)[1])
                
                mask = np.zeros(hyp_pic_calibrated.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, contours, 1)

                mask_applied = cv2.bitwise_and(color_mask, color_mask, mask=mask)
                pseudorgb_masked = cv2.addWeighted(false_rgb, 1 - transparency_level, mask_applied, transparency_level, 0)
                
                # This part is so repetitive, could be improved
                if batch == False:
                    for i, contour in enumerate(contours):
                        cv2.drawContours(pseudorgb_masked, [contour], -1, (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(contour)
                        centro_x = x + w // 2
                        centro_y = y + h // 2
                        cv2.putText(pseudorgb_masked, str(i), (centro_x, centro_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imwrite(os.path.join(segmented_pseudorgb_directory, f"segm_{pic}.png"), pseudorgb_masked)
                    
                    return contours
                else:
                    cv2.imwrite(os.path.join(segmented_pseudorgb_directory, f"segm_{pic}.png"), pseudorgb_masked)

                    return mask
        except:
            traceback.print_exc()
    else:
        print("Select a valid segmentation mode.")
        return None

