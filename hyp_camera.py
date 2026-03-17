"""
hyp_camera.py

Module for processing hyperspectral images processing

Main functionalities:
- Compression and decompression
- Pre-processing methods
- Functions to calibrate
- Class to process hyperspectral session

"""


import os
import re
import ast
import time
import json
import pickle
import cv2
import joblib
import lz4.frame
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
from joblib import Parallel, delayed
from scipy.ndimage import median_filter, convolve
from scipy.signal import savgol_filter
from skimage.morphology import remove_small_objects
import calibration as cl
import functions_processing as fp
from concurrent.futures import ThreadPoolExecutor
from exploration_modelling_functions import compute_dmodx

plt.ion()




def masked_smoothing(array, mask, size=23):
    """
    Apply a masked smoothing filter using convolution.

    This function performs local smoothing on a 2D array while considering a mask. 
    Only values inside the mask are included in the smoothing operation, and 
    normalization is applied to avoid bias in masked regions.

    Parameters
    ----------
    array : np.ndarray
        Input 2D array (e.g., an image).
    mask : np.ndarray
        Boolean or binary mask of the same shape as `array`. 
        True/1 values indicate pixels to include in the smoothing.
    size : int, optional (default=23)
        Size of the square kernel used for smoothing.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed version of the input array with masked normalization.
    """

    # Create a uniform square kernel for convolution
    kernel = np.ones((size, size), dtype=np.float32)

    # Apply mask: keep values inside the mask, set others to 0
    array_masked = np.where(mask, array, 0)

    # Convolve masked array to compute the local sum of valid values
    smoothed_sum = convolve(array_masked, kernel, mode='reflect')

    # Convolve the mask to count how many valid pixels contribute in each neighborhood
    valid_count = convolve(mask.astype(np.float32), kernel, mode='reflect')

    # Avoid division by zero in regions with no valid pixels
    valid_count[valid_count == 0] = 1

    # Normalize the sum by the number of valid pixels
    smoothed = smoothed_sum / valid_count

    return smoothed

def show_hyp_img(hyp_img, band1=100, band2=200, band3=300):
    """
    Display a hyperspectral image as an RGB composite.

    This function selects three spectral bands from a hyperspectral image, 
    normalizes them to 8-bit values (0–255), and displays the result as a 
    false-color RGB image. The image is resized depending on its dimensions 
    to allow easier visualization.

    Parameters
    ----------
    hyp_img : np.ndarray
        3D hyperspectral image (height x width x bands).
    band1 : int, optional (default=100)
        Band index used for the red channel.
    band2 : int, optional (default=200)
        Band index used for the green channel.
    band3 : int, optional (default=300)
        Band index used for the blue channel.

    Returns
    -------
    None
        Displays the RGB composite in a window using OpenCV.
    """

    # Normalize and convert selected bands to 8-bit format
    red_channel = cv2.normalize(
        hyp_img[:, :, band1].astype(np.float32), None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    green_channel = cv2.normalize(
        hyp_img[:, :, band2].astype(np.float32), None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    blue_channel = cv2.normalize(
        hyp_img[:, :, band3].astype(np.float32), None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Merge bands into an RGB image
    hyp_rgb = cv2.merge([red_channel, green_channel, blue_channel])

    # Resize the image depending on resolution for better visualization
    if hyp_rgb.shape[0] >= 4000:
        hyp_rgb = cv2.resize(hyp_rgb, (int(hyp_rgb.shape[1] * 0.15), 
                                       int(hyp_rgb.shape[0] * 0.15)))
    else:
        hyp_rgb = cv2.resize(hyp_rgb, (int(hyp_rgb.shape[1] * 0.25), 
                                       int(hyp_rgb.shape[0] * 0.25)))

    # Display the image in a window
    cv2.imshow("HYP_IMG", hyp_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_hyp_img_in_rgb(hyp_img, output_path, name, export=True, band1=1, band2=200, band3=20):
    """
    Generate an RGB composite from a hyperspectral image and optionally save it.

    This function selects three spectral bands from a hyperspectral image, 
    normalizes them to 8-bit values (0–255), and creates an RGB composite. 
    The image can either be saved to disk or returned as a NumPy array.

    Parameters
    ----------
    hyp_img : np.ndarray
        3D hyperspectral image (height x width x bands).
    output_path : str
        Path where the RGB image will be saved if `export=True`.
    name : str
        Name of the output file (without extension).
    export : bool, optional (default=True)
        If True, the image is saved to disk. If False, the image is returned as an array.
    band1 : int, optional (default=1)
        Band index used for the red channel.
    band2 : int, optional (default=200)
        Band index used for the green channel.
    band3 : int, optional (default=20)
        Band index used for the blue channel.

    Returns
    -------
    np.ndarray or None
        If `export=False`, returns the RGB composite as a NumPy array.
        If `export=True`, saves the image to disk and returns None.
    """

    # Normalize and convert selected bands to 8-bit format
    red_channel = cv2.normalize(
        hyp_img[:, :, band1].astype(np.float32), None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    green_channel = cv2.normalize(
        hyp_img[:, :, band2].astype(np.float32), None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    blue_channel = cv2.normalize(
        hyp_img[:, :, band3].astype(np.float32), None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Merge bands into an RGB image
    hyp_rgb = cv2.merge([red_channel, green_channel, blue_channel])

    if export:
        cv2.imwrite(f"{output_path}/{name}.jpg", hyp_rgb)
    else:
        return hyp_rgb

def compress_hyp_img(hyp_img, output_name, metadata=None, precision=16, num_threads=10):
    """
    Compress a hyperspectral image into LZ4 format and save metadata separately.

    The hyperspectral image is split into blocks, each compressed independently 
    in parallel to improve speed. Metadata including block information, shape, 
    and dtype is stored in a JSON file.

    Parameters
    ----------
    hyp_img : np.ndarray
        Hyperspectral image to be compressed (3D array).
    output_name : str
        Base name of the output files. Generates `output_name.lz4` and 
        `output_name.json`.
    metadata : dict, optional (default=None)
        Dictionary to store additional metadata. Will be updated with block 
        information automatically.
    precision : int, optional (default=16)
        Floating point precision to use: 16 (float16) or 32 (float32).
    num_threads : int, optional (default=10)
        Number of threads used for parallel compression.

    Returns
    -------
    None
        Saves the compressed data and metadata to disk.
    """
    if metadata is None:
        metadata = {}

    # Convert the image to chosen float precision
    dtype = np.float16 if precision == 16 else np.float32
    hyp_img = hyp_img.astype(dtype)

    # Split into blocks along axis 0 (rows)
    time_i_serial = time.time()
    blocks = np.array_split(hyp_img, num_threads, axis=0)
    time_f_serial = time.time()
    print(f"Serialization time: {time_f_serial - time_i_serial:.2f} seconds")

    block_metadata = []

    # Compress blocks in parallel
    time_i_compr = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        compressed_blocks = list(executor.map(
            lambda args: compress_block(*args),
            [(block, i, block_metadata) for i, block in enumerate(blocks)]
        ))
    time_f_compr = time.time()
    print(f"Compression time: {time_f_compr - time_i_compr:.2f} seconds")

    # Update metadata
    metadata["shape"] = hyp_img.shape
    metadata["dtype"] = str(dtype)
    metadata["num_blocks"] = len(blocks)
    metadata["blocks"] = block_metadata

    # Save metadata to JSON
    with open(f"{output_name}.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Save compressed blocks to binary file
    time_i_save = time.time()
    with open(f"{output_name}.lz4", "wb") as f:
        f.write(b"".join(compressed_blocks))
    time_f_save = time.time()
    print(f"Save time: {time_f_save - time_i_save:.2f} seconds")

    print("Compression process completed.")

def compress_block(block, index, block_metadata):
    """
    Compress a single data block and update metadata.

    Parameters
    ----------
    block : np.ndarray
        Sub-array (block) of the hyperspectral image to be compressed.
    index : int
        Block index in the sequence.
    block_metadata : list
        List to store metadata about each compressed block.

    Returns
    -------
    bytes
        Compressed block as bytes.
    """
    # Convert block to bytes
    block_bytes = block.tobytes()

    # Compress block using LZ4
    compressed_block = lz4.frame.compress(block_bytes)

    # Record metadata
    block_metadata.append({
        "index": index,
        "shape": block.shape,
        "dtype": str(block.dtype),
        "size": len(compressed_block)
    })

    return compressed_block

def decompress_block(block_data, block_shape, dtype):
    """
    Decompress a single block, cast to float32, and reshape to its original dimensions.

    Parameters
    ----------
    block_data : bytes
        Compressed block data.
    block_shape : tuple
        Original shape of the block before compression.
    dtype : str or np.dtype
        Data type used for the compressed block (e.g., np.float16 or np.float32).

    Returns
    -------
    np.ndarray
        Decompressed block as float32.
    """
    decompressed_block = lz4.frame.decompress(block_data)
    block = np.frombuffer(decompressed_block, dtype=dtype).reshape(block_shape)

    return block.astype(np.float32)

def decompress_hyp_img(hyp_path, array_shape=None, precision=16, num_threads=10):
    """
    Decompress a hyperspectral image from LZ4 format using metadata if available.

    Parameters
    ----------
    hyp_path : str
        Path to the `.lz4` file.
    array_shape : tuple, optional (default=None)
        Shape of the hyperspectral image if metadata is not available.
    precision : int, optional (default=16)
        Floating point precision used during compression: 16 (float16) or 32 (float32).
    num_threads : int, optional (default=10)
        Number of threads used for parallel decompression.

    Returns
    -------
    reconstructed_image : np.ndarray
        Decompressed hyperspectral image as float32.
    shape3d : tuple or None
        Original shape from metadata, or None if reconstructed using `array_shape`.
    """
    time_i_des = time.time()

    json_path = hyp_path.replace(".lz4", ".json")
    metadata = None

    # Try loading metadata if available
    try:
        print("Loading metadata")
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Metadata not found: {json_path}")
        metadata = None

    # Load compressed binary data
    with open(hyp_path, "rb") as f:
        compressed_data = f.read()

    if metadata:
        # Parallel decompression with metadata
        blocks_info = sorted(metadata["blocks"], key=lambda x: x["index"])
        shape3d = metadata["original_shape"]

        # Compute byte offsets for each block
        block_offsets = []
        start = 0
        for block in blocks_info:
            block_offsets.append((start, start + block["size"]))
            start += block["size"]

        dtype = np.float16 if precision == 16 else np.float32

        time_i_comp = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_block = {}

            for offset, block in zip(block_offsets, blocks_info):
                shape = tuple(block["shape"])  # Ensure tuple
                future = executor.submit(
                    decompress_block,
                    compressed_data[offset[0]:offset[1]],
                    shape,
                    dtype
                )
                future_to_block[future] = block["index"]

            # Collect results sorted by block index
            blocks = [future.result() for future in sorted(future_to_block, key=lambda x: future_to_block[x])]

        reconstructed_image = np.vstack(blocks)

        time_f_comp = time.time()
        print(f"Parallel decompression time: {time_f_comp - time_i_comp:.2f} seconds")

        return reconstructed_image, shape3d

    elif array_shape:
        # Sequential decompression without metadata
        pic_decompressed = lz4.frame.decompress(compressed_data)
        dtype = np.float16 if precision == 16 else np.float32
        reconstructed_image = np.frombuffer(pic_decompressed, dtype=dtype).reshape(array_shape).astype(np.float32)

    else:
        raise ValueError("No metadata found and `array_shape` not provided.")

    return reconstructed_image, None

def deadpixels_interpolation(hyp_pic):
    """
    Detect and interpolate dead pixels in a hyperspectral image.

    Dead pixels are defined as those with values < 1. If detected, they are 
    replaced using a median filter applied locally per band. This interpolation 
    helps reduce artifacts caused by defective sensor elements.

    Parameters
    ----------
    hyp_pic : np.ndarray
        Hyperspectral image (3D array: height x width x bands).

    Returns
    -------
    np.ndarray
        Hyperspectral image with dead pixels interpolated.
    """

    # Identify dead pixels (values < 1)
    deadpixels = hyp_pic < 1

    if np.any(deadpixels):
        print("Some pixels are dead")

        # Loop through each band (spectral dimension)
        for band in range(hyp_pic.shape[2]):
            if np.any(deadpixels[..., band]):
                print(f"Dead pixels found in band: {band}")

                # Apply median filter only where pixels are dead
                hyp_pic[..., band] = np.where(
                    deadpixels[..., band],
                    median_filter(hyp_pic[..., band], size=2),  # Interpolate with local median
                    hyp_pic[..., band]
                )
    else:
        print("No dead pixels detected")

    print("Pixels interpolated using median filter")
    return hyp_pic

def spikepixels_interpolation(hyp_pic, ntimes_threshold=6, filter_size=4):
    """
    Detect and interpolate spike pixels in a hyperspectral image.

    Spike pixels are defined as unusually high intensity values that exceed 
    a band-specific threshold (mean + N * std). Detected spikes are replaced 
    using a local median filter to reduce artifacts.

    Parameters
    ----------
    hyp_pic : np.ndarray
        Hyperspectral image (3D array: height x width x bands).
    ntimes_threshold : int or float, optional (default=6)
        Multiplier of the standard deviation used to define the spike threshold.
    filter_size : int, optional (default=4)
        Size of the median filter kernel used for interpolation.

    Returns
    -------
    np.ndarray
        Hyperspectral image with spike pixels interpolated.
    """

    # Compute mean and std per band to define spike threshold
    std_per_band = np.std(hyp_pic, axis=(0, 1))
    mean_per_band = np.mean(hyp_pic, axis=(0, 1))
    threshold_per_band = ntimes_threshold * std_per_band + mean_per_band

    # Broadcast threshold to match image dimensions
    threshold_per_band = np.tile(threshold_per_band, (hyp_pic.shape[0], hyp_pic.shape[1], 1))

    # Identify spike pixels
    spikespixels = hyp_pic > threshold_per_band

    if np.any(spikespixels):
        print("Some pixels are spikes")
        for band in range(hyp_pic.shape[2]):
            if np.any(spikespixels[..., band]):
                print(f"Spikes detected in band: {band}")

                # Apply median filter only at spike locations
                hyp_pic[..., band] = np.where(
                    spikespixels[..., band],
                    median_filter(hyp_pic[..., band], size=filter_size),
                    hyp_pic[..., band]
                )
    else:
        print("No spike pixels detected")

    print("Pixels interpolated using median filter")
    return hyp_pic

def hyp_snv(hyp_pic_2d):
    """
    Apply Standard Normal Variate (SNV) normalization to a 2D hyperspectral array.

    SNV is a row-wise normalization technique commonly used in spectroscopy 
    to remove scatter effects. Each row (spectrum) is centered by its mean 
    and scaled by its standard deviation.

    Parameters
    ----------
    hyp_pic_2d : np.ndarray
        2D hyperspectral data (rows = spectra, columns = wavelengths).

    Returns
    -------
    np.ndarray
        SNV-normalized 2D array of the same shape as input.
    """

    # Compute mean and standard deviation for each row
    mean_vals = np.mean(hyp_pic_2d, axis=1)
    std_vals = np.std(hyp_pic_2d, axis=1)

    # Apply SNV normalization: (row - mean) / std
    roi_array_SNV = (hyp_pic_2d - mean_vals[:, np.newaxis]) / std_vals[:, np.newaxis]

    return roi_array_SNV

def hyp_msc(hyp_pic_2d):
    """
    Apply Multiplicative Scatter Correction (MSC) to a 2D hyperspectral array.

    MSC is a preprocessing technique used to correct scatter effects in 
    spectral data. Each spectrum (row) is centered and scaled relative 
    to a reference spectrum (mean of all spectra), using a linear model.

    Parameters
    ----------
    hyp_pic_2d : np.ndarray
        2D hyperspectral data (rows = spectra, columns = wavelengths).

    Returns
    -------
    np.ndarray
        MSC-corrected 2D array of the same shape as input.
    """

    # 1️⃣ Center each spectrum by subtracting its mean
    row_means = hyp_pic_2d.mean(axis=1, keepdims=True)  # keepdims avoids using np.newaxis
    hyp_pic_2d_msc = hyp_pic_2d - row_means

    # 2️⃣ Compute reference spectrum (mean of all centered spectra)
    ref = hyp_pic_2d_msc.mean(axis=0)

    # 3️⃣ Fit linear model for all spectra at once (vectorized)
    # coefficients[0] = slope, coefficients[1] = intercept
    coefficients = np.polyfit(ref, hyp_pic_2d_msc.T, 1)

    # 4️⃣ Apply correction: normalize each spectrum using the fitted slope and intercept
    array2d_filtrado_msc = (hyp_pic_2d_msc.T - coefficients[1]) / coefficients[0]

    # 5️⃣ Restore original orientation
    hyp_pic_2d_msc = array2d_filtrado_msc.T

    return hyp_pic_2d_msc

def savisky(hyp_pic_2d, window, polyorder, deriv):
    """
    Apply Savitzky-Golay filter to 2D hyperspectral data.

    The Savitzky-Golay filter smooths spectra and can also compute derivatives.
    Filtering is applied row-wise (axis=1), which corresponds to the spectral dimension.

    Parameters
    ----------
    hyp_pic_2d : np.ndarray
        2D hyperspectral data (rows = spectra, columns = wavelengths).
    window : int
        Length of the filter window (must be odd and <= number of columns).
    polyorder : int
        Order of the polynomial used to fit the samples.
    deriv : int
        Order of the derivative to compute.

    Returns
    -------
    np.ndarray
        2D array of the same shape as input, filtered with Savitzky-Golay.
    """

    # Apply Savitzky-Golay filter row-wise
    hyp_pic_savgol = savgol_filter(
        hyp_pic_2d,
        window_length=window,
        polyorder=polyorder,
        axis=1,
        deriv=deriv
    )

    return hyp_pic_savgol

def compute_median(col):
    """
    Compute the median of a column or 1D array.

    This simple function is designed to be used in parallel processing,
    for example with joblib or concurrent.futures, to compute medians of 
    multiple columns independently.

    Parameters
    ----------
    col : np.ndarray or list
        1D array or list of numeric values.

    Returns
    -------
    float
        Median of the input values.
    """
    return np.median(col)

def obtain_hyp_df(hyp_pic, session, id, picture_name, n_element="bulk", median_calcul=True,
                  preproc_names=["RAW","SNV", "MSC", "SG1_W11_P2", "SG2_W11_P2", "SNV_SG1_W11_P2", "SNV_SG2_W11_P2"],
                   bands=[900,1750,2], save_preproc_array=False):
    
    """
    Generate a pandas DataFrame with statistical summaries (mean, median, std) 
    of a hyperspectral image after applying multiple preprocessing steps.

    Parameters
    ----------
    hyp_pic : np.ndarray
        Hyperspectral image (2D or 3D: rows x cols x bands).
    session : str
        Session identifier to add as metadata.
    id : str
        Sample identifier to add as metadata.
    picture_name : str
        Name of the picture to add as metadata.
    n_element : str, optional (default="bulk")
        Indicates if a bulk model has been used or elements are segmented separately.
    median_calcul : bool, optional (default=True)
        Whether to compute the median in addition to mean and std. It takes more time. If False, mean appears two times.
    preproc_names : list of str, optional
        List of preprocessing names to apply. Options include:
        RAW, SNV, MSC, SG1/SG2 variants, SNV+SG1/SG2, etc.
    bands : list of int, optional
        Band selection [start, stop, step] for the resulting DataFrame.
    save_preproc_array : bool, optional (default=False)
        Whether to save preprocessed arrays in a dictionary for further use. 
        If saving, avoid running many preprocessing methods to prevent exceeding memory.

    Returns
    -------
    df_merged : pd.DataFrame
        DataFrame containing mean, median, and std for each band and preprocessing step,
        plus metadata columns (Session, ID, Picture_name, n_element).
    preproc_array_dict : dict or None
        Dictionary with preprocessed arrays if `save_preproc_array=True`, else None.

    Notes
    -----
    The list below defines the spectral preprocessing types that can be applied:  
    - "RAW" refers to the unprocessed reflectance data.  
    - "SNV" (Standard Normal Variate) and "MSC" (Multiplicative Scatter Correction) are 
    commonly used normalization techniques to reduce scatter effects.  
    - "SG1_Wx_P2" indicates the application of the Savitzky-Golay filter (first derivative, SG1), 
    where "W" stands for the window size (e.g., W5 means a window of 5) and "P2" indicates a 2nd-order polynomial.  
    - Combinations such as "SNV_SG1_W5_P2" or "SG1_SNV_W5_P2" indicate whether SNV is applied 
    before or after the Savitzky-Golay filter.  

    All combinations listed can be used to evaluate which preprocessing method performs best for your analysis.

    """
    try:
        time_prep_i=time.time()

        # 1️⃣ Flatten 3D array to 2D (rows*cols x bands) if needed
        if len(hyp_pic.shape) == 3:
            hyp_pic_segment_flat = hyp_pic.reshape(-1, hyp_pic.shape[-1])  # From 3D (5008, 1280, 425) to 2D (5008*1280, 425)
            non_zero_rows_mask = np.any(hyp_pic_segment_flat != 0, axis=1)  # Remove zeros rows
            non_zero_indices = np.flatnonzero(non_zero_rows_mask) # Save nonzero index
            hyp_pic_segment_2d = hyp_pic_segment_flat[non_zero_indices] #Segment
            del(hyp_pic_segment_flat)
        else:
            hyp_pic_segment_2d = hyp_pic

        bands = np.arange(bands[0], bands[1], bands[2])

        time_prep_f=time.time()
        print(f"time presmothing {time_prep_f-time_prep_i}")
        results_dict = {"Band": bands}
        
        # For saving arrays
        preproc_array_dict = {}

        # 2️⃣ Loop through all preprocessing steps

        for preproc_name in preproc_names:
            
             # Apply RAW (no preprocessing)
            if preproc_name=="RAW":
                time_prep_i=time.time()
                preproc=hyp_pic_segment_2d
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")

            elif preproc_name=="SNV":
                time_prep_i=time.time()
                preproc=hyp_snv(hyp_pic_2d=hyp_pic_segment_2d)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
                
            elif preproc_name=="MSC":
                time_prep_i=time.time()
                preproc=hyp_msc(hyp_pic_2d=hyp_pic_segment_2d)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
            elif preproc_name.startswith("SG1"):
                time_prep_i=time.time()
                # Regular expression to obtain W and P values
                match = re.search(r"W(\d+)_P(\d+)", preproc_name)
                if match:
                    w_value = int(match.group(1))  # 'W'
                    p_value = int(match.group(2))  # 'P'
                preproc=savisky(hyp_pic_2d=hyp_pic_segment_2d, window=w_value, polyorder=p_value, deriv=1)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
            elif preproc_name.startswith("SG2") and "_SNV" not in preproc_name:
                time_prep_i=time.time()
                # Regular expression to obtain W and P values
                match = re.search(r"W(\d+)_P(\d+)", preproc_name)
                if match:
                    w_value = int(match.group(1))  # 'W'
                    p_value = int(match.group(2))  # 'P'
                preproc=savisky(hyp_pic_2d=hyp_pic_segment_2d, window=w_value, polyorder=p_value, deriv=2)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
            elif preproc_name.startswith("SNV_SG1"):
                time_prep_i=time.time()
                # Regular expression to obtain W and P values
                match = re.search(r"W(\d+)_P(\d+)", preproc_name)
                if match:
                    w_value = int(match.group(1))  # 'W'
                    p_value = int(match.group(2))  # 'P'
                preproc=hyp_snv(hyp_pic_2d=hyp_pic_segment_2d)
                preproc=savisky(hyp_pic_2d=preproc, window=w_value, polyorder=p_value, deriv=1)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
            elif preproc_name.startswith("SNV_SG2"):
                time_prep_i=time.time()
                # Regular expression to obtain W and P values
                match = re.search(r"W(\d+)_P(\d+)", preproc_name)
                if match:
                    w_value = int(match.group(1))  # 'W'
                    p_value = int(match.group(2))  # 'P'
                preproc=hyp_snv(hyp_pic_2d=hyp_pic_segment_2d)
                preproc=savisky(hyp_pic_2d=preproc, window=w_value, polyorder=p_value, deriv=2)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")

            elif preproc_name.startswith("SG1_SNV"):
                time_prep_i=time.time()
                # Regular expression to obtain W and P values
                match = re.search(r"W(\d+)_P(\d+)", preproc_name)
                if match:
                    w_value = int(match.group(1))  # 'W'
                    p_value = int(match.group(2))  # 'P'

                preproc=savisky(hyp_pic_2d=hyp_pic_segment_2d, window=w_value, polyorder=p_value, deriv=1)
                preproc=hyp_snv(hyp_pic_2d=preproc)
                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
            elif preproc_name.startswith("SG2_SNV"):
                time_prep_i=time.time()
                # Regular expression to obtain W and P values
                match = re.search(r"W(\d+)_P(\d+)", preproc_name)
                if match:
                    w_value = int(match.group(1))  # 'W'
                    p_value = int(match.group(2))  # 'P'

                preproc=savisky(hyp_pic_2d=hyp_pic_segment_2d, window=w_value, polyorder=p_value, deriv=2)
                preproc=hyp_snv(hyp_pic_2d=preproc)

                time_prep_f=time.time()
                print(f"time preparing {preproc_name} {time_prep_f-time_prep_i}")
            else:
                print("Preproc name no available")


            time_prep_i=time.time()
            # Meand and stdu to GPU
            preproc_gpu = cp.asarray(preproc)
            #non blocking True FAIL ERROR in the calculations # stream.synchronize()  # Only for nonblocking True
            stream = cp.cuda.Stream(non_blocking=False)
            with stream:
                mean_gpu = cp.mean(preproc_gpu, axis=0)
                std_gpu = cp.std(preproc_gpu, axis=0)
          
            mean = cp.asnumpy(mean_gpu)
            std = cp.asnumpy(std_gpu)

           
            time_prep_f=time.time()
            print(f"time post_smoothing meanstd -  {time_prep_f-time_prep_i}")
            
            if median_calcul:
                # 🏎️ Median calculation parallelize
                median = Parallel(n_jobs=10)(delayed(compute_median)(preproc[:, i]) for i in range(preproc.shape[1]))
                median = np.array(median)
            else:
                median=mean
            time_prep_f=time.time()

            print(f"time post_smoothing 1 -  {time_prep_f-time_prep_i}")

            time_prep_i=time.time()

            # Save in dictionary
            
            results_dict[f"Mean_{preproc_name}"] = mean
            results_dict[f"Median_{preproc_name}"] = median
            results_dict[f"Std_{preproc_name}"] = std

            time_prep_f=time.time()
            print(f"time post_smoothing 2 - {time_prep_f-time_prep_i}")

            # Save pre-processed array
            if save_preproc_array:
                preproc_array_dict[preproc_name] = preproc
                

            
        # 4️⃣ Dictionaries to df
        # Convertir a DataFrame
        df_merged = pd.DataFrame(results_dict)
        df_merged["Session"] = session
        df_merged["ID"] = id
        df_merged["Picture_name"] = picture_name
        df_merged["n_element"] = n_element
    except Exception as e:
        print(f"Ocurrió un error en picture")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje de error: {str(e)}")
        print("Detalle del error:")
                       
    if save_preproc_array:
        return df_merged, preproc_array_dict
    else:
        return df_merged, None

def save_cal_seg_hyppic(
    segment,
    path_session_1,
    hyp_pic,
    hyp_pic_name,
    calibration=True,
    batch_elements=True,
    n_element=None,
    element_row=None,
    non_zero_indices=None,
    original_shape=None
):
    """
    Save a hyperspectral image (HYP) after calibration and/or segmentation.

    Depending on the flags, the function saves the image in different folders:
    - CAL_SEG_BATCH: segmented batch of elements
    - CAL_SEG_INDV: segmented individual elements
    - CAL: calibrated full image

    Parameters
    ----------
    segment : bool
        If True, the hyperspectral image is segmented.
    path_session_1 : str
        Base path for the session where images will be saved.
    hyp_pic : np.ndarray
        Hyperspectral image array to save.
    hyp_pic_name : str
        Name of the hyperspectral image file.
    calibration : bool, optional (default=True)
        If True and segment=False, save the calibrated full image.
    batch_elements : bool, optional (default=True)
        If True, save as a batch of segmented elements.
    n_element : int or str, optional
        Element number or identifier for individual segmentation.
    element_row : pd.Series, optional
        Metadata row corresponding to the element (used when batch_elements=False).
    non_zero_indices : np.ndarray, optional
        Indices of non-zero pixels (used to reconstruct segmented images).
    original_shape : tuple, optional
        Original shape of the hyperspectral image.

    Returns
    -------
    pd.DataFrame or None
        If batch_elements=False and segment=True, returns a modified row with updated
        "Name_picture_HYP" and "element_number". Otherwise, returns None.
    """

    if segment and batch_elements:
        # Save segmented batch
        save_path = os.path.join(path_session_1, "HYP/CAL_SEG_BATCH/")
        os.makedirs(save_path, exist_ok=True)

        compress_hyp_img(
            hyp_img=hyp_pic,
            precision=16,
            output_name=os.path.join(save_path, hyp_pic_name),
            metadata={
                "original_shape": original_shape,
                "nonzero_indices": non_zero_indices.tolist(),
                "batch_elements": batch_elements,
                "n_element": "batch"
            }
        )

    elif segment and not batch_elements:
        # Save segmented individual element
        save_path = os.path.join(path_session_1, "HYP/CAL_SEG_INDV/")
        os.makedirs(save_path, exist_ok=True)

        compress_hyp_img(
            hyp_img=hyp_pic,
            precision=16,
            output_name=os.path.join(save_path, f"{hyp_pic_name}_{n_element}"),
            metadata={
                "original_shape": original_shape,
                "nonzero_indices": non_zero_indices.tolist(),
                "batch_elements": batch_elements,
                "n_element": n_element
            }
        )

        # Modify the metadata row for the individual element
        modified_row = element_row.copy()
        modified_row["Name_picture_HYP"] = f"{element_row['Name_picture_HYP']}_{n_element}"
        modified_row["element_number"] = n_element
        modified_row = modified_row.to_frame().T  # Convert Series to single-row DataFrame

        return modified_row

    elif calibration and not segment:
        # Save calibrated full image
        save_path = os.path.join(path_session_1, "HYP/CAL/")
        os.makedirs(save_path, exist_ok=True)

        compress_hyp_img(
            hyp_img=hyp_pic,
            precision=16,
            output_name=os.path.join(save_path, hyp_pic_name),
            metadata={"original_shape": original_shape}
        )


class hyp_session:
    
    """
    Class to manage hyperspectral imaging sessions, including loading session data, 
    calibrating hyperspectral images, segmenting samples, applying preprocessing, 
    and predicting traits using PLS or other models.

    This class centralizes workflows for handling hyperspectral images within a session, 
    from loading raw/reference data to generating processed datasets and predictions.

    Attributes
    ----------
    session : str
        Identifier of the current session.
    path_sessions : str
        Base path where all sessions are stored.
    path_session_1 : str
        Full path for the current session (path_sessions/session).
    results_directory : str
        Directory where processed results and outputs are stored.
    session_table : pd.DataFrame
        Table loaded from the session (metadata, file paths, references, etc.).
    quadratic_model : object
        Quadratic calibration model (if used).
    linear_model : object
        Linear calibration model (if used).
    calibration_type : str
        Type of calibration model used ("quadratic_3" or "linear_2").

    Methods
    -------
    load_table(filename="result_table_general.txt")
        Loads the session's metadata table from a text/CSV file.

    load_calibration_model(type="quadratic_3", pretrained=False, theor50_path=None, theor90_path=None)
        Loads or creates a calibration model (quadratic or linear) using reference hyperspectral images.

    calibrate_segment_preprocess(segment_type="manual_masks", band_index=50, masks_path=None, mask_color="red", 
                                transparency_level=0.5, segmented_pseudorgb_directory=None, save=True,
                                preproc=["RAW","SNV", "MSC", ...], median_cal=True, calibrate=True, segment=True,
                                subdir="HYP/RAW/", preprocessing=True, cut_head=0, cut_tail=None,
                                predict=False, obtain_pic_predicted=False, model_pls_df=None, bands=[900,1750,2],
                                pic_smoothing_px=10, **kwargs)
        Main pipeline method that:
        1. Decompresses hyperspectral images.
        2. Applies calibration (quadratic or linear).
        3. Segments the image using masks or other methods.
        4. Applies multiple spectral preprocessing techniques (RAW, SNV, MSC, SG filters, etc.).
        5. Computes statistical summaries (mean, median, std) per band.
        6. Saves processed hyperspectral images and metadata.
        7. Optionally predicts traits using pre-trained PLS or other models.

    Notes
    -----
    - Designed to handle large hyperspectral datasets with multiple samples in a session.
    - Supports both batch segmentation and individual element processing.
    - Integrates calibration, segmentation, preprocessing, and prediction in a single workflow.
    - Preprocessing options include: RAW, SNV, MSC, SG1/SG2 derivatives, SNV+SG, SG+SNV.
    - Predictions are made on selected bands using external models with optional scaling.
    """
    def __init__(self, session, path_sessions, results_directory):
        self.path_sessions=path_sessions
        self.session = session
        self.path_session_1=os.path.join(path_sessions, session)
        self.results_directory= results_directory

    def load_table(self, filename="result_table_general.txt"):
        """
    Load the session metadata table from a text or CSV file.

    This method reads a tab-delimited table containing metadata, file paths, 
    and reference information for the hyperspectral session. The table is stored 
    as a pandas DataFrame in the `session_table` attribute.

    Parameters
    ----------
    filename : str, optional
        Name of the table file to load. Default is "result_table_general.txt".
        Can be changed if the session table was split or regenerated.

    Notes
    -----
    - The method constructs the full path by combining the session path with 
      the filename.
    - If the file exists, it is loaded into `self.session_table`.
    - If the file does not exist, an error message is printed, and 
      `self.session_table` is not created.
    - Structure of the session file is provided in a info.txt file in the repository.
        """
        self.table_path = os.path.join(self.path_session_1, filename)

        if os.path.exists(self.table_path):
            # Read general table
            self.session_table = pd.read_csv(self.table_path, sep='\t')  
            print(f"File session {self.session} loaded")
        else:
            print(f"File session {self.session} - {self.table_path} error ")

    def load_calibration_model(self, type="quadratic_3", pretrained=False, theor50_path=None, theor90_path=None):

        """
        Load or create a calibration model for the hyperspectral session.

        Depending on the selected `type`, this method either loads a pre-trained model
        from a pickle file or builds a new calibration model using the reference hyperspectral images.

        Parameters
        ----------
        type : str, optional
            Type of calibration model to use. Options are:
            - "quadratic_3": three-point quadratic calibration using references at 0%, 50%, and 90%.
            - "linear_2": two-point linear calibration using references at 0% and 50%.
            Default is "quadratic_3".
        pretrained : bool, optional
            If True, load the model from a pickle file in the session folder. 
            If False, build the model from reference hyperspectral images. Default is False.
        theor50_path : str or None, optional
            Path to theoretical reference at 50%, used to build the model. Default is None.
        theor90_path : str or None, optional
            Path to theoretical reference at 90%, only used for "quadratic_3" calibration. Default is None.

        Notes
        -----
        - Reference images are loaded from the session table (`self.session_table`) and decompressed
        using `decompress_hyp_img`.
        - The model is saved as a pickle file (`quadratic_3_model.pkl` or `linear_2_model.pkl`) 
        in the session folder after creation.
        - For "quadratic_3", three reference images (0%, 50%, 90%) are used to create the model.
        - For "linear_2", two reference images (0%, 50%) are used.
        - If an unrecognized calibration type is provided, the method prints a warning and does nothing.
        """
        self.calibration_type=type

        if type == "quadratic_3":
            if pretrained:
                with open(os.path.join(self.path_session_1, "quadratic_3_model.pkl"), "rb") as f:
                    self.quadratic_model = pickle.load(f)
            else:
                ref_array_shape_list=self.session_table.at[0, 'Ref_Array_shape']
                ref_array_shape_list=ref_array_shape_list.strip("()").split(", ")
                ref_array_shape_list=[int(num) for num in ref_array_shape_list]


                #Obtain the paths of references
                path_0=self.session_table.at[0, 'Reference 0'] + ".lz4"
                file_name_ref_0 = os.path.basename(path_0)
                path_0=os.path.join(self.path_session_1, f"HYP/REFERENCES/{file_name_ref_0}")

                path_50=self.session_table.at[0, 'Reference 50'] + ".lz4"
                file_name_ref_50 = os.path.basename(path_50)
                path_50=os.path.join(self.path_session_1, f"HYP/REFERENCES/{file_name_ref_50}")
                
                path_90=self.session_table.at[0, 'Reference 90'] + ".lz4"
                file_name_ref_90 = os.path.basename(path_90)
                path_90=os.path.join(self.path_session_1, f"HYP/REFERENCES/{file_name_ref_90}")

                # Decompress references
                ref0=decompress_hyp_img(hyp_path=path_0, array_shape=ref_array_shape_list)[0]
                ref50=decompress_hyp_img(hyp_path=path_50, array_shape=ref_array_shape_list)[0]
                ref90=decompress_hyp_img(hyp_path=path_90, array_shape=ref_array_shape_list)[0]
                
                # Build quadratic model
                self.quadratic_model=cl.create_quadratic_model_calibration(ref0=ref0, ref50=ref50, ref90=ref90,
                                                                            theor50_path=theor50_path, theor90_path=theor90_path)
                print("Quadratic model calibration created for session:", self.session)

                with open(os.path.join(self.path_session_1, "quadratic_3_model.pkl"), "wb") as f:
                    pickle.dump(self.quadratic_model, f)

        elif type == "linear_2":
            if pretrained:
                with open(os.path.join(self.path_session_1, "linear_2_model.pkl"), "rb") as f:
                    self.linear_model = pickle.load(f)
            else:
                ref_array_shape_list=self.session_table.at[0, 'Ref_Array_shape']
                ref_array_shape_list=ref_array_shape_list.strip("()").split(", ")
                ref_array_shape_list=[int(num) for num in ref_array_shape_list]


                #Obtain the paths of references
                path_0=self.session_table.at[0, 'Reference 0'] + ".lz4"
                file_name_ref_0 = os.path.basename(path_0)
                path_0=os.path.join(self.path_session_1, f"HYP/REFERENCES/{file_name_ref_0}")

                path_50=self.session_table.at[0, 'Reference 50'] + ".lz4"
                file_name_ref_50 = os.path.basename(path_50)
                path_50=os.path.join(self.path_session_1, f"HYP/REFERENCES/{file_name_ref_50}")
                
                # Decompress references
                ref0=decompress_hyp_img(hyp_path=path_0, array_shape=ref_array_shape_list)[0]
                ref50=decompress_hyp_img(hyp_path=path_50, array_shape=ref_array_shape_list)[0]

                # Build quadratic model
                self.linear_model=cl.create_linear_model_calibration_only2(ref0=ref0, ref50=ref50,
                                                                            theor50_path=theor50_path)
                print("Linear model calibration created for session:", self.session)

                with open(os.path.join(self.path_session_1, "linear_2_model.pkl"), "wb") as f:
                    pickle.dump(self.linear_model, f)

        else:
            print("Calibration type selected no identified")

    def calibrate_segment_preprocess(self, segment_type="manual_masks", band_index=50, masks_path=None, mask_color="red", transparency_level=0.5,
                                      segmented_pseudorgb_directory=None, save=True,
                                        preproc=["RAW","SNV", "MSC", "SG1_W11_P2", "SG2_W11_P2", "SNV_SG1_W11_P2", "SNV_SG2_W11_P2"],
                                          median_cal=True, calibrate=True, segment=True, subdir="HYP/RAW/", preprocessing=True, cut_head=0, cut_tail=None,
                                           predict=False, obtain_pic_predicted=False,model_pls_df=None, bands=[900,1750,2], pic_smoothing_px=10, **kwargs ):
       
        """
        Process all hyperspectral images of a session by optionally performing calibration, segmentation, 
        spectral preprocessing, and prediction. Results, including preprocessed statistics and per-pixel 
        predicted heatmaps, are saved to disk.

        Parameters
        ----------
        segment_type : str, optional
            Type of segmentation to apply (e.g., 'manual_masks'). Default is 'manual_masks'.
        band_index : int, optional
            Band index used as reference for segmentation. Default is 50.
        masks_path : str or None, optional
            Path to manual masks if used. Default is None.
        mask_color : str, optional
            Color used for manual masks visualization. Default is 'red'.
        transparency_level : float, optional
            Transparency level of overlay when using colored masks. Default is 0.5.
        segmented_pseudorgb_directory : str or None, optional
            Directory to save segmented pseudo-RGB images. Default is None.
        save : bool, optional
            Whether to save calibrated and segmented hyperspectral images. Default is True.
        preproc : list of str, optional
            List of spectral preprocessing methods to apply (e.g., RAW, SNV, MSC, SG1/SG2). Default includes common methods.
        median_cal : bool, optional
            Whether to compute median in addition to mean and standard deviation. Default is True.
        calibrate : bool, optional
            Whether to apply calibration using the loaded calibration model. Default is True.
        segment : bool, optional
            Whether to segment the hyperspectral images. Default is True.
        subdir : str, optional
            Subdirectory of hyperspectral images relative to session path. Default is 'HYP/RAW/'.
        preprocessing : bool, optional
            Whether to execute spectral preprocessing. Default is True.
        cut_head : int, optional
            Number of bands to cut from the start of the spectrum. Default is 0.
        cut_tail : int or None, optional
            Number of bands to cut from the end of the spectrum. Default is None.
        predict : bool, optional
            Whether to apply prediction models for traits. Default is False.
        obtain_pic_predicted : bool, optional
            Whether to generate per-pixel predicted heatmaps. Default is False.
        model_pls_df : str or None, optional
            Path to CSV file containing prediction models and settings. Default is None.
        bands : list of int, optional
            Spectral range [start, end, step] used for preprocessing. Default is [900, 1750, 2].
        pic_smoothing_px : int, optional
            Pixel size for smoothing predicted heatmaps. Default is 10.
        **kwargs : dict
            Additional parameters passed to segmentation or preprocessing functions.

        Returns
        -------
        None
            All outputs are written to disk:
            - Preprocessed statistics per image in a tab-separated file.
            - Predicted trait values per image in a tab-separated file.
            - Per-pixel prediction heatmaps (if `obtain_pic_predicted=True`).
            - Error log with images that could not be processed.

        Notes
        -----
        - The function iterates over all images in the session table, decompresses them, and applies calibration, 
        segmentation, and preprocessing steps in order.
        - Segmentation filters out pixels with zero values, and preprocessing applies spectral methods and band selection.
        - Prediction can be performed using pre-trained models and optionally scaled according to configuration.
        - Heatmaps are rescaled, smoothed, and saved as RGB images with trait values annotated.
        - Any errors during processing are logged and do not stop the function from processing the remaining images.
        """
                
    
        error_list=[]
        if predict:
            df_predict_results = pd.DataFrame(columns=['Session', 'ID', 'Picture_name','Preprocessing', 'Trait', 'Trait_value', 'DModX'])
        
        # Loop over the pictures of the session

        for index, row in self.session_table.iterrows():
            try:
                # Obtain pic info
                hyp_pth, array_shape, sample_name, pic = fp.process_picture_info(row, self.session, self.path_session_1, subdir=subdir)
                
                #Hyp decompress
                hyp_pic=decompress_hyp_img(hyp_path=hyp_pth, array_shape=array_shape, precision=16)[0]

               
                # Execute reference calibration
                if calibrate:
                    if self.calibration_type == "quadratic_3":
                        hyp_pic=cl.deploy_quadratic_model_calibration(hyp_picture=hyp_pic, 
                                                                            coefficient_a=self.quadratic_model[0], coefficient_b=self.quadratic_model[1], coefficient_c=self.quadratic_model[2])
                    elif self.calibration_type=="linear_2":
                        hyp_pic=cl.deploy_linear_model_calibration(hyp_picture=hyp_pic, 
                                                                            coefficient_m=self.linear_model[0], coefficient_b=self.linear_model[1])
                    
                
                # Execute segmentation
                if segment:

                    mask_gray_2d=fp.segment_image(segment_type=segment_type, 
                    pic=pic, 
                    hyp_pic_calibrated=hyp_pic, 
                    band_index=band_index, 
                    masks_path=masks_path, 
                    mask_color=mask_color, 
                    transparency_level=transparency_level, 
                    segmented_pseudorgb_directory=segmented_pseudorgb_directory, **kwargs)

                    # 1️⃣ Flatten the array (convert it to 2D)
                    hyp_pic = hyp_pic.reshape(-1, hyp_pic.shape[-1])

                    mask_gray_2d = mask_gray_2d.astype(np.uint8).reshape(-1, 1)

                    # 2️⃣ Identify non-zero rows (at least one value different from zero in their columns)
                    non_zero_rows_mask = np.any(mask_gray_2d != 0, axis=1)  # axis=1 checks row by row

                    # 3️⃣ Store the indices of the non-zero rows
                    non_zero_indices = np.flatnonzero(non_zero_rows_mask)  # Indices of rows that are not completely zeros

                    # 4️⃣ Filter the flattened array using the indices of non-zero rows
                    hyp_pic = hyp_pic[non_zero_indices]
                else:
                    non_zero_indices=None

                    
                # Save calibrated and optionally segmented pic
                if save == True:
                    save_cal_seg_hyppic(segment=segment, path_session_1=self.path_session_1, hyp_pic=hyp_pic, calibration=calibrate,
                                                                batch_elements=True, hyp_pic_name=pic, original_shape=array_shape,
                                                                non_zero_indices=non_zero_indices)

                # Execute pre-processing methods
                if preprocessing:
                    time_smooth_i=time.time()
                    # Apply cut
                    hyp_pic = hyp_pic[:, cut_head:cut_tail]
                    # Compute number of original bands
                    start_nm, end_nm, step_nm = bands
                    original_num_bands = (end_nm - start_nm) // step_nm  # e.g., 425
                    # Convert negative cut_tail to positive index (from the beginning)
                    if cut_tail < 0:
                        cut_tail = original_num_bands + cut_tail  # e.g., 425 - 40 = 385
                    # Compute new spectral range
                    new_start_nm = start_nm + cut_head * step_nm
                    new_end_nm = start_nm + cut_tail * step_nm
                    # Update bands
                    bands = [new_start_nm, new_end_nm, step_nm]
                    # Print updated info
                    print(f"Updated bands: {bands}")  # e.g., [950, 1668, 2]
                    
                    # Spectral pre-processing
                    df_pic_results=obtain_hyp_df(hyp_pic=hyp_pic,  session=self.session, id=sample_name, picture_name=pic,
                                                n_element="bulk", preproc_names=preproc, median_calcul=median_cal, bands=bands,
                                                        save_preproc_array=obtain_pic_predicted)
                    if 'df_all_results' not in locals():
                        df_all_results = df_pic_results[0]
                    else:
                        df_all_results = pd.concat([df_all_results, df_pic_results[0]], ignore_index=True)
                    time_smooth_f=time.time()
                    print("Time_smoothing", time_smooth_f-time_smooth_i)
                
                # Execute prediction models
                if predict:
                    model_df = pd.read_csv(model_pls_df, sep="\t")

                    # Iterate over models
                    for _, row in model_df.iterrows():
                        trait = row["Trait"]
                        metric_forgeneral_result = row["Metric"]
                        metric = row["Metric"].replace("Mean_", "").replace("Median_", "")
                        model_path = row["model_path"]
                        scale_path = row.get("scale_path", None)
                        heatmap_scale = row.get("heatmap_scale", None)
                        bands_selected = row.get('Bands_selected', None)
                        print("Trait: ",trait)

                        # Bulk mean or median spectra of the picture for the general result
                        bulk_array_result = df_pic_results[0][[metric_forgeneral_result]].to_numpy().reshape(1, -1)
                        
                        # Load model
                        model = joblib.load(model_path)
                            # Sclae if there is scale
                        if pd.notna(scale_path) and scale_path != "":
                            scaler = joblib.load(scale_path)
                            bulk_array_result = scaler.transform(bulk_array_result)
                        else:
                            bulk_array_result = bulk_array_result

                        # Parse Bands_selected si es string
                        if isinstance(bands_selected, str):
                            try:
                                bands_selected_general = ast.literal_eval(bands_selected)
                                bulk_array_result = bulk_array_result[:, bands_selected_general]

                            except Exception as e:
                                print(f"⚠️ Parding error for {trait} ({metric}): {e}")
                                bands_selected_general = None

                        # Predict general result
                        predicted_value = model.predict(bulk_array_result)[0]
                        print(f"Trait: {trait} -> {predicted_value}")
                        # Compute DModX
                        dmodx = compute_dmodx(model, bulk_array_result)[0]


                        # Save result
                        df_predict_results.loc[len(df_predict_results)] = {
                                                                            'Session': self.session,
                                                                            'ID': sample_name,
                                                                            'Picture_name': pic,
                                                                            'Preprocessing': metric_forgeneral_result,
                                                                            'Trait': trait,
                                                                            'Trait_value': predicted_value,
                                                                            'DModX':dmodx
                                                                            
                                                                        }

                        # To obtain the heatmap pic of the trait 
                        if obtain_pic_predicted:
                            # Obtain corresponding array
                            array = df_pic_results[1].get(metric)
                            if array is None:
                                print(f"⚠️ Metric '{metric}' not found in df_pic_results[1], ommiting{trait}")
                                continue

                            # Scale
                            if pd.notna(scale_path) and scale_path != "":
                                scaler = joblib.load(scale_path)
                                array_scaled = scaler.transform(array)
                            else:
                                array_scaled = array

                            
                            # Parse Bands_selected
                            if isinstance(bands_selected, str):
                                try:
                                    bands_selected = ast.literal_eval(bands_selected)
                                    array_scaled = array_scaled[:, bands_selected]

                                except Exception as e:
                                    print(f"⚠️ No se pudo parsear Bands_selected para {trait} ({metric}): {e}")
                                    bands_selected = None

                            # Predict all the pixels 
                            prediction = model.predict(array_scaled)
                            
                            # Reescale values to 0-255 using references of config file
                            if pd.notna(heatmap_scale) and "_" in heatmap_scale:
                                try:
                                    esc_min, esc_max = map(float, heatmap_scale.split("_"))
                                    prediction_rescaled = (prediction - esc_min) / (esc_max - esc_min) * 255
                                    prediction_rescaled = np.clip(prediction_rescaled, 0, 255).astype(np.uint8)
                                except Exception as e:
                                    print(f"⚠️ Error in the scale {trait}: {e}")
                                    prediction_rescaled = np.zeros_like(prediction, dtype=np.uint8)
                            else:
                                prediction_rescaled = np.zeros_like(prediction, dtype=np.uint8)

                            # Dictionary to store only the 2D arrays
                            full_pred_arrays_2d = {}

                            # Extract the original dimensions (height and width)
                            height, width = array_shape[:2]

                            # Initialize an array filled with -500 for the total size
                            full_array = np.full(height * width, -500, dtype=np.float32)

                            # Insert the predicted values into the valid indices
                            full_array[non_zero_indices] = prediction_rescaled

                            # Reshape to 2D
                            full_array_2d = full_array.reshape((height, width))

                            # Mask of valid values (different from -500)
                            valid_mask_2d = full_array_2d != -500

                            # Apply smoothing without affecting the -500 values
                            smoothed_array_2d = masked_smoothing(full_array_2d, valid_mask_2d, size=pic_smoothing_px)

                            # Restore -500 in invalid regions if you want to keep it
                            smoothed_array_2d[~valid_mask_2d] = -500

                            # Save the smoothed result
                            full_pred_arrays_2d[trait] = smoothed_array_2d

                            colormap = cm.get_cmap("jet")

                            for trait, array_2d in full_pred_arrays_2d.items():
                                trait_folder = os.path.join(self.results_directory, trait)
                                
                                # If the trait folder does not exist, create it
                                if not os.path.exists(trait_folder):
                                    os.makedirs(trait_folder)
                                
                                # Create a mask for valid values (different from -500)
                                valid_mask = array_2d != -500

                                # Create an empty RGB image (all black initially)
                                rgba_img = np.zeros((*array_2d.shape, 3), dtype=np.uint8)

                                # Normalize only the valid values between 0 and 255
                                norm = colors.Normalize(vmin=0, vmax=255)
                                normalized = norm(array_2d[valid_mask])

                                # Apply colormap and convert to uint8
                                colored = (colormap(normalized)[:, :3] * 255).astype(np.uint8)

                                # Fill only the valid regions
                                rgba_img[valid_mask] = colored

                                # Define text and position
                                text = f"{trait}: {predicted_value:.2f}" 
                                position = (100, rgba_img.shape[0] - 100)  # 100 px from the left, 100 px from the bottom

                                # Put the text in white
                                cv2.putText(rgba_img,
                                            text,
                                            position,
                                            cv2.FONT_HERSHEY_SIMPLEX,  # font type
                                            1,                          # font scale
                                            (255, 255, 255),            # white color in BGR
                                            2,                          # thickness
                                            cv2.LINE_AA)                # anti-aliased
                                                                # Define text and position
                                # text = f"DModX: {dmodx:.2f}" 
                                # position = (100, rgba_img.shape[0] - 50)  # 100 px from the left, 100 px from the bottom

                                # # Put the text in white
                                # cv2.putText(rgba_img,
                                #             text,
                                #             position,
                                #             cv2.FONT_HERSHEY_SIMPLEX,  # font type
                                #             1,                          # font scale
                                #             (255, 255, 255),            # white color in BGR
                                #             2,                          # thickness
                                #             cv2.LINE_AA)                # anti-aliased
                                
                                # bar_legend

                                # bar_legend
                                height, width, _ = rgba_img.shape

                                bar_height = 20
                                text_bar_height = 50

                                norm = colors.Normalize(vmin=0, vmax=255)
                                colormap = cm.get_cmap("jet")

                                # Crear barra de colores
                                bar_array = np.linspace(0, 255, width)
                                bar_rgb = (colormap(norm(bar_array))[:, :3] * 255).astype(np.uint8)
                                bar_rgb = np.tile(bar_rgb[np.newaxis, :, :], (bar_height, 1, 1))

                                # Crear fondo negro para los números
                                text_bar = np.zeros((text_bar_height, width, 3), dtype=np.uint8)

                                # Dibujar escala basada en esc_min y esc_max
                                if pd.notna(heatmap_scale) and "_" in heatmap_scale:

                                    esc_min, esc_max = map(float, heatmap_scale.split("_"))

                                    # 10 divisiones
                                    values = np.linspace(esc_min, esc_max, 11)

                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    font_scale = 0.6
                                    thickness = 1

                                    for v in values:

                                        pos_ratio = (v - esc_min) / (esc_max - esc_min)
                                        x = int(pos_ratio * width)

                                        text = f"{v:.2f}".rstrip("0").rstrip(".")

                                        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

                                        # centrar texto
                                        x_text = x - text_w // 2

                                        # evitar que se salga de la imagen
                                        x_text = max(0, min(x_text, width - text_w))

                                        y_text = int(text_bar_height * 0.75)

                                        cv2.putText(
                                            text_bar,
                                            text,
                                            (x_text, y_text),
                                            font,
                                            font_scale,
                                            (255,255,255),
                                            thickness,
                                            cv2.LINE_AA
                                        )

                                # Unir todo
                                final_img = np.vstack([rgba_img, text_bar, bar_rgb])
                                # Save the image
                                filename = f"{trait}_{os.path.basename(pic)}.png"
                                save_path = os.path.join(trait_folder, filename)
                                Image.fromarray(final_img).save(save_path)



            except Exception as e:
                # Catch any error that occurs during the processing of the image `pic`
                # and store the error information in `error_list` for later review.
                print(f"An error occurred in picture {pic}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Error details:")
                error_list.append([pic, e])
                # Continue with the next image without stopping the entire processing
                continue

        # Export results
        if preprocessing:
            df_all_results.to_csv(f'{self.results_directory}/all_results_{self.session}.txt', sep='\t', index=False)
        if predict:
            df_predict_results.to_csv(f'{self.results_directory}/predicted_results_{self.session}.txt', sep='\t', index=False)

        df_errors = pd.DataFrame(error_list, columns=[ "Picture","Error_Message"])
        df_errors.to_csv(f'{self.results_directory}/error_logs_{self.session}.txt', sep='\t', index=False)


