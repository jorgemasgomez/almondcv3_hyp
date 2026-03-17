"""
calibration.py

This script contains functions for spectral calibration of hyperspectral images. 
It provides pixelwise calibration models for line-scan images, supporting linear 
and quadratic approaches using reference measurements at different reflectance levels.

The main purpose of these functions is to compute calibration coefficients that 
correct raw hyperspectral data based on known reference panels, allowing accurate 
reflectance estimation across the spectral range.
"""
import numpy as np
import pandas as pd
import time
import cupy as cp

def train_quadratic_model_calibration(ref, theor):
    """
    Fits a quadratic model to the given reflectance data.

    This function computes the coefficients of a second-degree polynomial (quadratic) 
    that best fits the measured reflectance values (`ref`) to the theoretical reference values (`theor`).

    Parameters
    ----------
    ref : numpy.ndarray
        Measured reflectance values. Should be a 1D array of length n_samples.
    theor : numpy.ndarray
        Theoretical reference reflectance values corresponding to `ref`. Should be a 1D array of the same length.

    Returns
    -------
    tuple of float
        Quadratic coefficients (a, b, c) such that:
            theor ≈ a * ref**2 + b * ref + c

    Notes
    -----
    This function uses `numpy.polyfit` with degree 2 to fit the polynomial.
    """
    
    coefficient_a, coefficient_b, coefficient_c= np.polyfit(ref, theor, 2)
    return coefficient_a, coefficient_b, coefficient_c

def create_quadratic_model_calibration(ref0, ref50, ref90, theor50_path,
                                       theor90_path, band_i=900, band_f=1750, band_step=2, n_bands=425):
    """
    Creates pixelwise quadratic calibration models for line-scan hyperspectral images.

    This function calculates quadratic coefficients (a, b, c) for each pixel column and spectral band
    using measured reflectance values at 0%, 50%, and 90% references, along with theoretical reference values.
    Calibration is applied pixelwise for each line of the hyperspectral image. Each column is calibrated individually
    based on sensor response and known white references.

    The method is based on:
    Burger, J., & Geladi, P. (2005). Hyperspectral NIR image regression part I: calibration and correction. 
    Journal of Chemometrics, 19(5‐7), 355-363.

    Two calibration types are supported:
    - "linear_2": Linear calibration using two points (0% and 50% reflectance), fitting a straight line.
    - "quadratic_3": Quadratic calibration using three points (0%, 50%, 90% reflectance), fitting a second-degree polynomial
      to correct for sensor non-linearity. This function implements the quadratic_3 approach.

    Reference Reflectance File Structure
    -----------------------------------
    The CSV file contains spectral reference data, typically for a standard white or reference panel.
    It has two columns, separated by commas:
        1. nm  : Wavelength in nanometers (numeric)
        2. R   : Reflectance value at that wavelength (numeric, usually 0-100)
    
    Example:
        nm,R
        250,51.540694
        251,51.505076

    Parameters
    ----------
    ref0 : numpy.ndarray
        Measured spectral reflectance at 0% reference (dark). Shape: (n_samples, n_columns, n_bands).
    ref50 : numpy.ndarray
        Measured spectral reflectance at 50% reference. Shape: (n_samples, n_columns, n_bands).
    ref90 : numpy.ndarray
        Measured spectral reflectance at 90% reference. Shape: (n_samples, n_columns, n_bands).
    theor50_path : str
        Path to CSV file containing theoretical reflectance values for 50% reference.
    theor90_path : str
        Path to CSV file containing theoretical reflectance values for 90% reference.
    band_i : int, optional
        Initial wavelength to consider (nm). Default is 950.
    band_f : int, optional
        Final wavelength to consider (nm). Default is 1750.
    band_step : int, optional
        Step between consecutive wavelengths (nm). Default is 2.
    n_bands : int, optional
        Number of bands to use. Default is 425.

    Returns
    -------
    tuple of numpy.ndarray
        Three arrays containing the quadratic coefficients (a, b, c) for each pixel column and band.
        Each array has shape (1, n_columns*n_samples_per_column, n_bands).

    Notes for Developers
    --------------------
    This function can be optimized: it currently relies on nested for-loops over columns and bands.
    Vectorization or parallelization could significantly improve performance for large hyperspectral images.

    """
    # Select the spectral bands of interest using the new parameters
    bands = np.arange(band_i, band_f, band_step)

    # Create the theoretical reflectance for 0% (dark reference)
    theor0 = np.zeros(n_bands)

    # Load theoretical reflectance values for 50% reference from CSV and filter by the selected bands
    theor50 = pd.read_csv(theor50_path)
    theor50 = theor50[theor50.iloc[:, 0].isin(bands)].iloc[:, 1].values


    # Load theoretical reflectance values for 90% reference from CSV and filter by the selected bands
    theor90 = pd.read_csv(theor90_path)
    theor90 = theor90[theor90.iloc[:, 0].isin(bands)].iloc[:, 1].values

    
    # Initialize arrays to store quadratic coefficients (a, b, c) for each pixel and band
    coefficients_a_array = np.empty((ref50.shape[1] * ref50.shape[2]), dtype=np.float32)
    coefficients_b_array = np.empty((ref50.shape[1] * ref50.shape[2]), dtype=np.float32)
    coefficients_c_array = np.empty((ref50.shape[1] * ref50.shape[2]), dtype=np.float32)

    count = 0
    nrows = ref50.shape[0]

    # Loop over each pixel column and spectral band to fit the quadratic model
    for column in range(ref50.shape[1]):
        for band in range(theor0.shape[0]):
            # Stack theoretical values for 0%, 50%, and 90% references and repeat for each sample
            theor = np.stack((
                np.repeat(theor0[band], nrows),
                np.repeat(theor50[band], nrows),
                np.repeat(theor90[band], nrows)
            ))
            theor = theor.reshape(-1, 1)  # Ensure the theoretical array is column-shaped

            # Concatenate measured reflectance for 0%, 50%, and 90% references for the current pixel and band
            ref = np.concatenate((ref0[:, column, band], ref50[:, column, band], ref90[:, column, band]), axis=0)
            ref = ref.ravel()  # Flatten the array to 1D for fitting

            # Fit a quadratic model to the measured vs theoretical reflectance
            coefficient_a, coefficient_b, coefficient_c = train_quadratic_model_calibration(ref, theor)

            # Store the coefficients in the pre-allocated arrays
            coefficients_a_array[count] = coefficient_a
            coefficients_b_array[count] = coefficient_b
            coefficients_c_array[count] = coefficient_c
            count += 1

    # Reshape the coefficient arrays to (n_columns*n_samples_per_column, n_bands)
    coefficients_a_array = coefficients_a_array.reshape(1280, 425)
    coefficients_b_array = coefficients_b_array.reshape(1280, 425)
    coefficients_c_array = coefficients_c_array.reshape(1280, 425)

    # Add a new axis to match the expected output shape (1, n_columns*n_samples_per_column, n_bands)
    coefficients_a_array = coefficients_a_array[np.newaxis, :, :]
    coefficients_b_array = coefficients_b_array[np.newaxis, :, :]
    coefficients_c_array = coefficients_c_array[np.newaxis, :, :]

    # Return the arrays of quadratic coefficients
    return coefficients_a_array, coefficients_b_array, coefficients_c_array

def deploy_quadratic_model_calibration(hyp_picture, coefficient_a, coefficient_b, coefficient_c,
                                        reserved_mem_gb=2, narrays_temporales=10):
    """
    Applies pixelwise quadratic calibration to a hyperspectral image using GPU (CuPy).

    This function performs a pixelwise quadratic correction for a hyperspectral line-scan image.
    Calibration is applied in batches to manage GPU memory usage, converting arrays between CPU and GPU as needed.

    Parameters
    ----------
    hyp_picture : numpy.ndarray
        Raw hyperspectral image data. Shape: (n_samples, n_columns, n_bands).
    coefficient_a : numpy.ndarray
        Quadratic coefficient 'a' for calibration. Shape must match image dimensions.
    coefficient_b : numpy.ndarray
        Linear coefficient 'b' for calibration. Shape must match image dimensions.
    coefficient_c : numpy.ndarray
        Constant coefficient 'c' for calibration. Shape must match image dimensions.
    reserved_mem_gb : float, optional
        Amount of GPU memory to reserve for other processes (GB). Default is 2.
    narrays_temporal : int, optional
        Number of temporary arrays considered for GPU memory estimation.
        Larger values reserve more memory and reduce batch size (safer, but more batches).
        Smaller values use less memory, allowing larger batches (faster, but risk of GPU memory overflow).
        Default is 10.

    Returns
    -------
    numpy.ndarray
        Calibrated hyperspectral image in CPU memory, same shape as `hyp_picture`.

    Notes for Developers
    --------------------
    - The function currently uses explicit for-loops for batch processing.
    - GPU memory is managed manually, and arrays are converted between NumPy and CuPy.
    - Performance could be improved with full vectorization or better memory management.
    """

    # Initialize an empty array to store the calibrated image (CPU memory)
    hyp_picture_calibrated = np.empty_like(hyp_picture)

    # Start timing the conversion of arrays to GPU
    time_i_cp_conversion = time.time()

    # Convert the hyperspectral image and coefficients to GPU arrays
    hyp_picture = cp.asarray(hyp_picture)
    coefficient_a = cp.asarray(coefficient_a)
    coefficient_b = cp.asarray(coefficient_b)
    coefficient_c = cp.asarray(coefficient_c)

    time_f_cp_conversion = time.time()
    print(f"Time to convert arrays to GPU: {time_f_cp_conversion - time_i_cp_conversion:.2f} seconds")

    # Get current GPU device
    device = cp.cuda.Device(0)

    # Retrieve GPU memory information
    total_mem, free_mem = device.mem_info

    # Reserve some memory for other processes
    free_mem -= reserved_mem_gb * 1024**3
    free_mem_gb = free_mem / (1024 ** 3)
    print(f"Available GPU memory after reserving {reserved_mem_gb} GB: {free_mem_gb:.2f} GB")

    # Estimate the size of a single line in memory
    dtype_size = np.dtype(np.float32).itemsize
    hyp_picture_size = hyp_picture.shape[1] * hyp_picture.shape[2] * dtype_size * narrays_temporales
    n_samples = hyp_picture.shape[0]

    # Calculate batch size based on available GPU memory
    batch_size = int(free_mem / hyp_picture_size)
    batch_number = int(n_samples / batch_size)
    print(f"Calculated batch size: {batch_size} lines, number of batches: {batch_number}")

    # Start batch processing
    time_i_serial = time.time()
    for start in range(0, n_samples, batch_size):
        batch_start_time = time.time()
        end = min(start + batch_size, n_samples)

        # Extract the current batch
        hyp_batch = hyp_picture[start:end]

        # Perform quadratic calibration on GPU
        square_hyp_batch = cp.square(hyp_batch)               # Square the image values
        term1_gpu = cp.multiply(coefficient_a, square_hyp_batch)  # Quadratic term
        del square_hyp_batch                                    # Free memory

        term2_gpu = cp.multiply(hyp_batch, coefficient_b)      # Linear term
        del hyp_batch                                           # Free memory

        # Sum quadratic, linear, and constant terms
        hyp_picture_calibrated_batch = cp.add(cp.add(term1_gpu, term2_gpu), coefficient_c)
        del term1_gpu, term2_gpu                                # Free memory

        # Move the calibrated batch back to CPU
        time_i_np_conversion = time.time()
        hyp_picture_calibrated_batch = cp.asnumpy(hyp_picture_calibrated_batch)
        time_f_np_conversion = time.time()
        cp._default_memory_pool.free_all_blocks()
        print(f"Time to convert batch back to NumPy: {time_f_np_conversion - time_i_np_conversion:.2f} seconds")

        # Store calibrated batch in the final array
        hyp_picture_calibrated[start:end] = hyp_picture_calibrated_batch

        batch_end_time = time.time()
        print(f"Time to process batch {start}-{end}: {batch_end_time - batch_start_time:.2f} seconds")

    time_f_serial = time.time()
    print(f"Total processing time (all batches): {time_f_serial - time_i_serial:.2f} seconds")

    return hyp_picture_calibrated

def train_linear_model_calibration(ref, theor):
    """
    Fits a linear calibration model between measured and theoretical reflectance.

    This function calculates the slope (m) and intercept (b) of a linear model
    that maps measured reflectance values to theoretical reference values.
    It is typically used for 2-point linear calibration (0% and 50% reflectance).

    Parameters
    ----------
    ref : numpy.ndarray
        Measured reflectance values (1D array).
    theor : numpy.ndarray
        Theoretical reflectance values corresponding to `ref` (1D array).

    Returns
    -------
    tuple of float
        - coefficient_m : float
            Slope of the linear calibration model.
        - coefficient_b : float
            Intercept of the linear calibration model.
    """
    
    # Fit a linear polynomial (degree 1) to measured vs theoretical reflectance
    coefficient_m, coefficient_b = np.polyfit(ref, theor, 1)
    
    return coefficient_m, coefficient_b

def create_linear_model_calibration_only2(ref0, ref50, theor50_path,
                                          band_i=900, band_f=1750, band_step=2, n_bands=425):
    """
    Creates pixelwise linear calibration models using 2 reference points (0% and 50% reflectance)
    for line-scan hyperspectral images.

    This function calculates linear coefficients (slope `m` and intercept `b`) for each pixel column
    and spectral band using measured reflectance at 0% and 50%, along with theoretical reference values.
    Calibration is applied pixelwise for each line of the hyperspectral image.

    Parameters
    ----------
    ref0 : numpy.ndarray
        Measured spectral reflectance at 0% reference (dark). Shape: (n_samples, n_columns, n_bands).
    ref50 : numpy.ndarray
        Measured spectral reflectance at 50% reference. Shape: (n_samples, n_columns, n_bands).
    theor50_path : str
        Path to CSV file containing theoretical reflectance values for 50% reference.
    band_i : int, optional
        Starting wavelength (nm) for calibration. Default is 900.
    band_f : int, optional
        Ending wavelength (nm) for calibration. Default is 1750.
    band_step : int, optional
        Step size between spectral bands (nm). Default is 2.
    n_bands : int, optional
        Total number of spectral bands. Default is 425.

    Returns
    -------
    tuple of numpy.ndarray
        Two arrays containing the linear coefficients (m, b) for each pixel column and band.
        Each array has shape (1, n_columns*n_samples_per_column, n_bands).

    Notes for Developers
    --------------------
    - This function currently relies on nested for-loops over columns and bands.
    - Vectorization or parallelization could improve performance for large hyperspectral images.
    """

    # Select the spectral bands of interest based on user-defined range and step
    bands = np.arange(band_i, band_f, band_step)

    # Create the theoretical reflectance for 0% (dark reference)
    theor0 = np.zeros(n_bands)

    # Load theoretical 50% reflectance from CSV and filter by selected bands
    theor50 = pd.read_csv(theor50_path)
    theor50 = theor50[theor50.iloc[:, 0].isin(bands)].iloc[:, 1].values

    # Initialize arrays to store linear coefficients
    coefficients_m_array = np.empty((ref50.shape[1]*ref50.shape[2]), dtype=np.float32)
    coefficients_b_array = np.empty((ref50.shape[1]*ref50.shape[2]), dtype=np.float32)

    n_samples = ref50.shape[0]
    counter = 0

    # Loop over columns and spectral bands (pixelwise calibration)
    for column in range(ref50.shape[1]):
        for band in range(n_bands):
            # Stack theoretical 0% and 50% reflectance
            theor = np.stack((np.repeat(theor0[band], n_samples), np.repeat(theor50[band], n_samples)))
            theor = theor.reshape(-1, 1)

            # Concatenate measured reflectances for 0% and 50%
            ref = np.concatenate((ref0[:, column, band], ref50[:, column, band]), axis=0)
            ref = ref.ravel()  # Ensure the array is 1D

            # Train linear calibration model
            coefficient_m, coefficient_b = train_linear_model_calibration(ref, theor)
            coefficients_m_array[counter] = coefficient_m
            coefficients_b_array[counter] = coefficient_b
            counter += 1

    # Reshape arrays to match original image dimensions
    coefficients_m_array = coefficients_m_array.reshape(ref50.shape[1], n_bands)
    coefficients_b_array = coefficients_b_array.reshape(ref50.shape[1], n_bands)

    # Add new axis to match output convention (1, n_columns, n_bands)
    coefficients_m_array = coefficients_m_array[np.newaxis, :, :]
    coefficients_b_array = coefficients_b_array[np.newaxis, :, :]

    return coefficients_m_array, coefficients_b_array

def deploy_linear_model_calibration(hyp_picture, coefficient_m, coefficient_b,
                                   reserved_mem_gb=2, narrays_temporal=10):
    """
    Applies pixelwise linear calibration to a hyperspectral image using GPU (CuPy).

    This function performs a pixelwise linear correction for a hyperspectral line-scan image.
    Calibration is applied in batches to manage GPU memory usage, converting arrays between CPU and GPU as needed.

    Parameters
    ----------
    hyp_picture : numpy.ndarray
        Raw hyperspectral image data. Shape: (n_samples, n_columns, n_bands).
    coefficient_m : numpy.ndarray
        Linear coefficient (slope) for each pixel column and band. Shape must match image dimensions.
    coefficient_b : numpy.ndarray
        Linear coefficient (intercept) for each pixel column and band. Shape must match image dimensions.
    reserved_mem_gb : float, optional
        Amount of GPU memory to reserve for other processes (GB). Default is 2.
    narrays_temporal : int, optional
        Number of temporary arrays considered for GPU memory estimation.
        Larger values reserve more memory and reduce batch size (safer, but more batches).
        Smaller values use less memory, allowing larger batches (faster, but risk of GPU memory overflow).
        Default is 10.
    Returns
    -------
    numpy.ndarray
        Calibrated hyperspectral image in CPU memory, same shape as `hyp_picture`.

    Notes for Developers
    --------------------
    - The function currently uses explicit for-loops for batch processing.
    - GPU memory is managed manually, and arrays are converted between NumPy and CuPy.
    - Performance could be improved with full vectorization or better memory management.
    """

    # Initialize an empty array to store the calibrated image (CPU memory)
    hyp_picture_calibrated = np.empty_like(hyp_picture)

    # Start timing the conversion of arrays to GPU
    time_i_cp_conversion = time.time()

    # Convert the hyperspectral image and coefficients to GPU arrays
    hyp_picture = cp.asarray(hyp_picture)
    coefficient_m = cp.asarray(coefficient_m)
    coefficient_b = cp.asarray(coefficient_b)

    time_f_cp_conversion = time.time()
    print(f"Time to convert arrays to GPU: {time_f_cp_conversion - time_i_cp_conversion:.2f} seconds")

    # Get current GPU device
    device = cp.cuda.Device(0)

    # Retrieve GPU memory information
    total_mem, free_mem = device.mem_info

    # Reserve some memory for other processes
    free_mem -= reserved_mem_gb * 1024**3
    free_mem_gb = free_mem / (1024 ** 3)
    print(f"Available GPU memory after reserving {reserved_mem_gb} GB: {free_mem_gb:.2f} GB")

    # Estimate the size of a single line in memory
    dtype_size = np.dtype(np.float32).itemsize
    hyp_picture_size = hyp_picture.shape[1] * hyp_picture.shape[2] * dtype_size * narrays_temporal
    n_samples = hyp_picture.shape[0]

    # Calculate batch size based on available GPU memory
    batch_size = int(free_mem / hyp_picture_size)
    batch_number = int(n_samples / batch_size)
    print(f"Calculated batch size: {batch_size} lines, number of batches: {batch_number}")

    # Start batch processing
    time_i_serial = time.time()
    for start in range(0, n_samples, batch_size):
        batch_start_time = time.time()
        end = min(start + batch_size, n_samples)

        # Extract the current batch
        hyp_batch = hyp_picture[start:end]

        # Apply linear calibration on GPU: slope * value
        term_gpu = cp.multiply(hyp_batch, coefficient_m)
        del hyp_batch  # Free memory

        # Add intercept
        hyp_picture_calibrated_batch = cp.add(term_gpu, coefficient_b)
        del term_gpu  # Free memory

        # Move the calibrated batch back to CPU
        time_i_np_conversion = time.time()
        hyp_picture_calibrated_batch = cp.asnumpy(hyp_picture_calibrated_batch)
        time_f_np_conversion = time.time()
        print(f"Time to convert batch back to NumPy: {time_f_np_conversion - time_i_np_conversion:.2f} seconds")

        # Store calibrated batch in the final array
        hyp_picture_calibrated[start:end] = hyp_picture_calibrated_batch

        batch_end_time = time.time()
        print(f"Time to process batch {start}-{end}: {batch_end_time - batch_start_time:.2f} seconds")

    time_f_serial = time.time()
    print(f"Total processing time (all batches): {time_f_serial - time_i_serial:.2f} seconds")

    return hyp_picture_calibrated