import zipfile
import os
import shutil
from ultralytics import YOLO, settings
import torch
import cv2
import numpy as np
import yaml
from sahi.predict import get_sliced_prediction,  AutoDetectionModel
from sahi.slicing import slice_image
from PIL import Image
settings.update({"wandb":False})


import os
import cv2
import numpy as np
from ultralytics import YOLO
from multiprocessing import Process, Queue



class ModelSegmentation():
    def __init__(self, working_directory):
        """
        Initializes the model with the specified working directory and sets the device to either CPU or GPU.
        
        Parameters:
            working_directory (str): The directory where the input data and results will be stored.
        
        Returns:
            None
        """
        self.working_directory = working_directory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory
            gpu_memory_gb = gpu_memory / (1024 ** 3)
            print(f"Detected GPU: {gpu_name}")
            print(f"Total GPU Memory: {gpu_memory_gb:.2f} GB")
        else:
            print("No GPU detected. Using CPU.")

    def train_segmentation_model(self, input_zip, pre_model="yolov8n-seg.pt", epochs=100, imgsz=640, batch=-1, name_segmentation="",
                                 retina_masks=True, colab=False):
        """
        Trains the segmentation model using a YOLO segmentation file.

        Parameters:
            input_zip (str): Path to the input zip file containing images and annotations.
            pre_model (str): Path to the pretrained model (default: "yolov8n-seg.pt").
            epochs (int): Number of epochs for training (default: 100).
            imgsz (int): Image size for training (default: 640).
            batch (int): Batch size for training (default: -1, auto batch size).
            name_segmentation (str): Name for the segmentation model's output (default: "").
            retina_masks (bool): Whether to use retina masks for segmentation (default: True).
            
            

        Returns:
            results_test_set (list): List of results containing the predicted masks for the test set.
        """
        input_zip_no_extension, extension = os.path.splitext(input_zip)
        output_folder_zip = os.path.join(self.working_directory, input_zip_no_extension)
        self.output_folder_zip = output_folder_zip

        if os.path.exists(self.output_folder_zip):
            shutil.rmtree(self.output_folder_zip)
        os.makedirs(self.output_folder_zip, exist_ok=True)

        zip_file_path = os.path.join(self.working_directory, input_zip)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_folder_zip)

        yaml_file = os.path.join(self.output_folder_zip, "data.yaml")
        self.yaml_file = yaml_file

        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        modified_data = {
            'path': self.output_folder_zip,
            'train': 'images/Train',
            'val': 'images/Validation',
            'test': 'images/Test'
        }
        if 'names' in data:
            modified_data['names'] = data['names']

        with open(self.yaml_file, 'w') as file:
            yaml.dump(modified_data, file, default_flow_style=False)

        results_models_directory = os.path.join(self.working_directory, f"results_models_segmentation_{name_segmentation}")
        self.results_models_directory = results_models_directory
        os.makedirs(self.results_models_directory, exist_ok=True)

        if colab:
            model = YOLO(pre_model)
            model.to(self.device)
            model.train(data=self.yaml_file, epochs=epochs, imgsz=imgsz, batch=batch, project=name_segmentation, name="results_training")
            shutil.move(name_segmentation, results_models_directory)
            
            test_set_folder = os.path.join(self.output_folder_zip, "images/Test/")
            self.test_set_folder = test_set_folder

            results_test_set = model.predict(self.test_set_folder, imgsz=imgsz, show=False, save=True, show_boxes=False, project=self.results_models_directory, save_txt=True,
                                            name="predictions_test", retina_masks=retina_masks)
        else:
            model = YOLO(pre_model)
            model.to(self.device)
            model.train(data=self.yaml_file, epochs=epochs, imgsz=imgsz, batch=batch, project=self.results_models_directory, name="results_training")

            test_set_folder = os.path.join(self.output_folder_zip, "images/Test/")
            self.test_set_folder = test_set_folder

            results_test_set = model.predict(self.test_set_folder, imgsz=imgsz, show=False, save=True, show_boxes=False, project=self.results_models_directory, save_txt=True,
                                            name="predictions_test", retina_masks=retina_masks)
        return results_test_set

    def predict_model(self, model_path, folder_input, imgsz=640, check_result=False, conf=0.6, max_det=300, retina_mask=True):
        """
        Predicts segmentation masks using a trained model.

        Parameters:
            model_path (str): Path to the trained model file.
            folder_input (str): Path to the folder containing images to be segmented.
            imgsz (int): Image size for prediction (default: 640).
            check_result (bool): Whether to save the results for further inspection (default: False).
            conf (float): Confidence threshold for predictions (default: 0.6).
            max_det (int): Maximum number of detections per image (default: 300).
            retina_mask (bool): Whether to use retina masks for segmentation (default: True).

        Returns:
            results (list): List of predictions for each image in the input folder. Each entry contains masks and coordinates of segmented objects.
        """
        model = YOLO(model_path)
        if not check_result:
            results = model.predict(folder_input, imgsz=imgsz, show=False, save=False, show_boxes=False, conf=conf, max_det=max_det, retina_masks=retina_mask)
        else:
            results = model.predict(folder_input, imgsz=imgsz, show=False, save=True, show_boxes=False, project=self.working_directory, name="check_results", conf=conf, max_det=max_det, retina_masks=retina_mask)

        return results

    def predict_model_sahi(self, model_path, folder_input=None, confidence_treshold=0.5, model_type='yolov8',
                            slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS", check_result=False
                            , postprocess_match_metric="IOS", postprocess_match_threshold=0.5, retina_masks=True, imgsz=640, image_array=None):
        """
        Predicts segmentation masks using the SAHI method (Slice and Heal Inference) for large images.

        Parameters:
            model_path (str): Path to the trained model file.
            folder_input (str): Path to the folder containing images to be segmented.
            image_array : Option for a direct picture array
            confidence_treshold (float): Confidence threshold for predictions (default: 0.5).
            model_type (str): Type of YOLO model to use (default: "yolov8").
            slice_height (int): Height of each slice (default: 640).
            slice_width (int): Width of each slice (default: 640).
            overlap_height_ratio (float): Overlap ratio in height between slices (default: 0.2).
            overlap_width_ratio (float): Overlap ratio in width between slices (default: 0.2).
            postprocess_type (str): Type of postprocessing for results (default: "NMS").
            check_result (bool): Whether to save the results for inspection (default: False).
            postprocess_match_metric (str): Metric to use for postprocessing (default: "IOS").
            postprocess_match_threshold (float): Threshold for postprocessing (default: 0.5).
            retina_masks (bool): Whether to use retina masks (default: True).
            imgsz (int): Image size for prediction (default: 640).

        Returns:
            results_list (list): List of results for each image processed, including segmented masks and additional information.
        """
        detection_model_seg = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_treshold,
            device=self.device,
            retina_masks=retina_masks,
            image_size=imgsz)
        
        if folder_input is not None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_list = [os.path.join(folder_input, file)
                        for file in os.listdir(folder_input)
                        if os.path.splitext(file)[1].lower() in image_extensions]
        elif image_array is not None:
            image_list = [image_array]
        else:
            raise ValueError("Provide a folder or a picture")

        results_list = []
        i = 1
        for pic in image_list:
            print(f"Pic {i}/{len(image_list)}")

            try:
                result = get_sliced_prediction(
                    image=pic, detection_model=detection_model_seg, slice_height=slice_height,
                    slice_width=slice_width, overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio,
                    postprocess_type=postprocess_type, postprocess_match_metric=postprocess_match_metric,
                    postprocess_match_threshold=postprocess_match_threshold, perform_standard_pred=True)
            except Exception as e:
                print(f"Error processing segmentation image {pic}: {e}")
                continue

            torch.cuda.empty_cache()
            results_list.append([result, pic])
            i += 1
            if check_result:
                pic_sin_ext = os.path.splitext(os.path.basename(pic))[0]
                check_result_path = os.path.join(self.working_directory, "check_results")
                os.makedirs(check_result_path, exist_ok=True)
                result.export_visuals(export_dir=check_result_path, hide_labels=True, rect_th=1, file_name=f"prediction_result_{pic_sin_ext}")
        return results_list

    # def slice_predict_reconstruct(self, imgsz, model_path, slice_width, slice_height, overlap_height_ratio, overlap_width_ratio,  input_folder=None, conf=0.5, retina_mask=True, image_array=None):
    #     """
    #     Slices large images, performs segmentation predictions on each slice, and reconstructs the full mask.

    #     Parameters:
    #         input_folder (str): Path to the folder containing the images to be sliced and processed.
    #         image_array : Option for a direct picture array
    #         imgsz (int): Image size for prediction (default: 640).
    #         model_path (str): Path to the trained model.
    #         slice_width (int): Width of each slice.
    #         slice_height (int): Height of each slice.
    #         overlap_height_ratio (float): Overlap ratio in height between slices.
    #         overlap_width_ratio (float): Overlap ratio in width between slices.
    #         conf (float): Confidence threshold for predictions (default: 0.5).
    #         retina_mask (bool): Whether to use retina masks (default: True).

    #     Returns:
    #         mask_list_images (list): List of reconstructed masks for each image processed.
    #     """
    #     if input_folder is not None:
    #         image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    #         image_list = [os.path.join(input_folder, file)
    #                       for file in os.listdir(input_folder)
    #                       if os.path.splitext(file)[1].lower() in image_extensions]
    #     elif image_array is not None:
    #         image_list = [image_array]
    #     else:
    #         raise ValueError("Provide a folder or a picture")

    #     mask_list_images = []
    #     n = 1
    #     for image_path in image_list:

    #         if image_array is None:
    #             print(f"Image {n}/{len(image_list)}")
    #             image_selected = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #         else:
    #             image_selected=image_path



    #         if image_selected.shape[2] == 4:
    #             image_selected = cv2.cvtColor(image_selected, cv2.COLOR_RGBA2RGB)

    #         # Slice the image into smaller chunks
    #         image_sliced = slice_image(image=image_selected, slice_width=slice_width,
    #                                    slice_height=slice_height, overlap_height_ratio=overlap_height_ratio,
    #                                    overlap_width_ratio=overlap_width_ratio, verbose=True)

    #         slice_count = 0
    #         mask_complete = np.zeros((image_sliced.original_image_height, image_sliced.original_image_width), dtype=np.uint8)

    #         # Process each slice
    #         for slice in image_sliced.images:
    #             model = YOLO(model_path, verbose=False)
    #             model.to(self.device)
    #             results = model.predict(slice, imgsz=imgsz, show=False, save=False, show_boxes=False,
    #                                     verbose=False, conf=conf, retina_masks=retina_mask)

    #             h_slice = slice.shape[0]
    #             w_slice = slice.shape[1]

    #             # Initialize an empty mask for the current slice
    #             mask_combined_slice = np.zeros((h_slice, w_slice), dtype=np.uint8)

    #             for result in results:
    #                 if result is None or result.masks is None or result.masks.data is None:
    #                     continue  # Skip to next iteration if no mask is present

    #                 # For each mask in the result, combine them into the slice's mask
    #                 for j, mask in enumerate(result.masks.data):
    #                     mask = mask.cpu().numpy() * 255  # Convert mask to 0-255 scale
    #                     mask = cv2.resize(mask, (w_slice, h_slice))  # Resize the mask to the slice size
    #                     mask_combined_slice = cv2.bitwise_or(mask_combined_slice, mask.astype(np.uint8))  # Combine the masks

    #             # Place the mask back into the complete image mask at the correct position
    #             mask_added=np.zeros((image_sliced.original_image_height, image_sliced.original_image_width), dtype=np.uint8)
    #             start_x=image_sliced.starting_pixels[slice_count][0]
    #             start_y=image_sliced.starting_pixels[slice_count][1]
    #             mask_added[start_y:start_y + h_slice, start_x:start_x + w_slice] = mask_combined_slice
    #             mask_complete = cv2.bitwise_or(mask_complete, mask_added)
    #             slice_count=slice_count+1

    #         # Save or return the full mask for the current image
    #         mask_list_images.append([mask_complete,image_path])
    #         n += 1

    #     return mask_list_images


    def slice_predict_reconstruct(self, imgsz, model_path, slice_width, slice_height, overlap_height_ratio, 
                                overlap_width_ratio, input_folder=None, conf=0.5, retina_mask=True, image_array=None):
        """
        Slices images, runs segmentation on all slices in batch (GPU), and reconstructs the full mask.
        """
        if input_folder is not None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_list = [os.path.join(input_folder, file)
                        for file in os.listdir(input_folder)
                        if os.path.splitext(file)[1].lower() in image_extensions]
        elif image_array is not None:
            image_list = [image_array]
        else:
            raise ValueError("Provide a folder or a picture")

        mask_list_images = []
        n = 1

        # Cargar modelo una sola vez
        model = YOLO(model_path, verbose=False).to(self.device)

        for image_path in image_list:
            if image_array is None:
                print(f"Processing Image {n}/{len(image_list)}")
                image_selected = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            else:
                image_selected = image_path

            if image_selected.shape[2] == 4:
                image_selected = cv2.cvtColor(image_selected, cv2.COLOR_RGBA2RGB)

            # Slice the image into chunks
            image_sliced = slice_image(image=image_selected, slice_width=slice_width,
                                    slice_height=slice_height, overlap_height_ratio=overlap_height_ratio,
                                    overlap_width_ratio=overlap_width_ratio, verbose=True)
            mask_complete = np.zeros((image_sliced.original_image_height, image_sliced.original_image_width), dtype=np.uint8)
            # Si image_sliced.images es una lista de arrays de NumPy
            slices_batch = image_sliced.images  # Ya están en formato NumPy

            # Aseguramos que se pase sin conversión a tensores
            with torch.no_grad():  # Evita cálculos innecesarios de gradientes
                results = model.predict(slices_batch, imgsz=imgsz, conf=conf, retina_masks=retina_mask, verbose=False)

            for i, result in enumerate(results):
                h_slice, w_slice = image_sliced.images[i].shape[:2]
                start_x, start_y = image_sliced.starting_pixels[i]

                mask_combined_slice = np.zeros((h_slice, w_slice), dtype=np.uint8)

                if result.masks and result.masks.data is not None:
                    for mask in result.masks.data:
                        mask = mask.cpu().numpy() * 255
                        mask = cv2.resize(mask, (w_slice, h_slice))
                        mask_combined_slice = cv2.bitwise_or(mask_combined_slice, mask.astype(np.uint8))

                mask_added = np.zeros_like(mask_complete)
                mask_added[start_y:start_y + h_slice, start_x:start_x + w_slice] = mask_combined_slice
                mask_complete = cv2.bitwise_or(mask_complete, mask_added)

            mask_list_images.append([mask_complete, image_path])
            n += 1

        return mask_list_images