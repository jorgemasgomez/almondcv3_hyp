import subprocess
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

def install_morphometrics_packages_r():
    # Path to the R script file that will install the packages
    r_script_path = r'Install_morphometrics.R'  # Make sure the path is correct

    # Command to execute the R script
    command = ['Rscript', r_script_path]

    # Execute the command using subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Show the output of the R command
        print("R command output:")
        print(result.stdout)

        # Show any error if it occurs
        if result.stderr:
            print("Error:")
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error while executing the R script: {e.stderr}")


def exploratory_morphometrics_r(
    info_data, 
    grouping_factor, 
    input_directory, 
    output_directory,
    img_width_panel=750, img_height_panel=500, 
    img_width_stack=750, img_height_stack=500,
    nexamples=1, 
    nharmonics=10, 
    img_width_ptolemy=750, img_height_ptolemy=500,
    img_width_deviations=750, img_height_deviations=500,
    img_width_reconstructions=750, img_height_reconstructions=500,
    show=True
):
    """
    Executes the R script with the provided arguments and optionally displays the generated images.

    Parameters:
    - info_data (str): Path to the data file with the information to be used.
    - grouping_factor (str): Name of the grouping column in info_data.
    - input_directory (str): Directory containing the .jpg images.
    - img_width_panel (int): Width of the image for the panel plot.
    - img_height_panel (int): Height of the image for the panel plot.
    - img_width_stack (int): Width of the image for the stack plot.
    - img_height_stack (int): Height of the image for the stack plot.
    - nexamples (int): Number of examples for the loop in the R script.
    - nharmonics (int): Number of harmonics for the calibration functions.
    - img_width_ptolemy (int): Width of the image for the Ptolemy plot.
    - img_height_ptolemy (int): Height of the image for the Ptolemy plot.
    - img_width_deviations (int): Width of the image for the efourier deviations plot.
    - img_height_deviations (int): Height of the image for the efourier deviations plot.
    - img_width_reconstructions (int): Width of the image for the efourier reconstructions plot.
    - img_height_reconstructions (int): Height of the image for the efourier reconstructions plot.
    - show (bool): If True, it will display the generated images using matplotlib.
    """
    
    # Fixed path to the R script
    script_r_path = "Exploratory_analysis.R"
    
    # Create the command to execute the R script with the arguments
    command = [
        'Rscript', 
        script_r_path,
        info_data, 
        grouping_factor, 
        input_directory, 
        str(img_width_panel), 
        str(img_height_panel), 
        str(img_width_stack), 
        str(img_height_stack), 
        output_directory,  
        str(nexamples), 
        str(nharmonics), 
        str(img_width_ptolemy), 
        str(img_height_ptolemy), 
        str(img_width_deviations), 
        str(img_height_deviations), 
        str(img_width_reconstructions), 
        str(img_height_reconstructions)
    ]
    
    # Execute the command using subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Show the output of the R command
        print("R command output:")
        print(result.stdout)

        # Show any errors if they occur
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # If 'show' is True, attempt to display the generated images
        if show:
            # Define the paths of the images exported by R
            exploratory_plots_dir = os.path.join(output_directory, 'exploratory_plots')
            panel_image_path = os.path.join(exploratory_plots_dir, 'panel_output.png')
            stack_image_path = os.path.join(exploratory_plots_dir, 'stack_output.png')
            
            # Show the panel and stack images
            if os.path.exists(panel_image_path):
                img = mpimg.imread(panel_image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            
            if os.path.exists(stack_image_path):
                img = mpimg.imread(stack_image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()

            # Show images generated in the loop from R (Ptolemy, deviations, reconstructions)
            for i in range(1, nexamples + 1):
                ptolemy_image_path = os.path.join(exploratory_plots_dir, f"ptolemy_output_{i}.png")
                deviations_image_path = os.path.join(exploratory_plots_dir, f"deviations_efourier_output_{i}.png")
                reconstructions_image_path = os.path.join(exploratory_plots_dir, f"reconstructions_efourier_output_{i}.png")
                
                # Check and display each image generated in the loop
                for image_path in [ptolemy_image_path, deviations_image_path, reconstructions_image_path]:
                    if os.path.exists(image_path):
                        img = mpimg.imread(image_path)
                        plt.imshow(img)
                        plt.axis('off')
                        plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error while executing the R script: {e.stderr}")


def run_efourier_pca_morphometrics_r(path_outline_objects, nharmonics, output_directory, 
                                      img_width_boxplot=1000, img_height_boxplot=1000, 
                                      img_width_pca=1000, img_height_pca=1000, show=False, 
                                      normalize="FALSE", start_point="FALSE", align_x="TRUE"):
    """
    Executes the R script "efourier_morphometrics.R" with the provided arguments.

    Parameters:
    - path_outline_objects (str): Path to the RDS file containing the outlines.
    - nharmonics (int): Number of harmonics for Fourier analysis.
    - output_directory (str): Output directory where the results will be saved.
    - img_width_boxplot (int): Width of the boxplot image.
    - img_height_boxplot (int): Height of the boxplot image.
    - img_width_pca (int): Width of the PCA plot image.
    - img_height_pca (int): Height of the PCA plot image.
    - show (bool): If True, shows the generated PCA plot using matplotlib.
    - normalize (str): Whether to normalize the data, "FALSE" or "TRUE".
    - start_point (str): Whether to set the start point, "FALSE" or "TRUE".
    - align_x (str): Whether to align the x-axis, "TRUE" or "FALSE".
    """

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Path to the R script
    script_r_path = "efourier_morphometrics.R"

    # Create the command to execute the R script with the provided arguments
    command = [
        'Rscript', 
        script_r_path, 
        str(path_outline_objects),  # Path to the RDS file with outlines
        str(nharmonics),            # Number of harmonics
        str(output_directory),      # Output directory
        str(img_width_boxplot),     # Boxplot image width
        str(img_height_boxplot),    # Boxplot image height
        str(img_width_pca),         # PCA image width
        str(img_height_pca),        # PCA image height
        str(normalize),             # Normalize the data
        str(start_point),           # Set the start point
        str(align_x)                # Align the x-axis
    ]

    # Execute the command using subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Show the output of the R command
        print("R command output:")
        print(result.stdout)

        # Show any errors if they occur
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # If 'show' is True, attempt to display the PCA plot
        if show:
            pca_image_path = os.path.join(output_directory, "efourier_results", "pca_output.png")
            if os.path.exists(pca_image_path):
                img = mpimg.imread(pca_image_path)
                plt.imshow(img)
                plt.axis('off')  # Turn off the axes
                plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error while executing the R script: {e.stderr}")


def run_plot_pca_morphometrics_r(input_directory, output_directory, img_width_pca=1000, img_height_pca=1000, 
                                 grouping_factor="", PC_axis1=1, PC_axis2=2, 
                                 chull_layer="FALSE", chullfilled_layer="FALSE", show=True):
    """
    Executes the R script "plot_pca_morphometrics.R" with the provided arguments.

    Parameters:
    - input_directory (str): Path to the directory containing the PCA object.
    - output_directory (str): Output directory where the results will be saved.
    - img_width_pca (int): Width of the PCA plot image.
    - img_height_pca (int): Height of the PCA plot image.
    - grouping_factor (str): Optional grouping factor for the PCA visualization.
    - PC_axis1 (int): Primary axis for PCA in the plot (default is 1).
    - PC_axis2 (int): Secondary axis for PCA in the plot (default is 2).
    - chull_layer (str): If "TRUE", adds a convex hull layer.
    - chullfilled_layer (str): If "TRUE", adds a filled convex hull layer.
    - show (bool): If True, shows the generated PCA plot using matplotlib.
    """

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Path to the R script
    script_r_path = "plot_pca_morphometrics.R"

    # Create the command to execute the R script with the provided arguments
    command = [
        'Rscript', 
        script_r_path, 
        str(input_directory),      # Path to the directory with the PCA object
        str(output_directory),      # Output directory
        str(img_width_pca),         # PCA image width
        str(img_height_pca),        # PCA image height
        str(grouping_factor),       # Optional grouping factor
        str(PC_axis1),              # PC1 axis
        str(PC_axis2),              # PC2 axis
        str(chull_layer),           # Convex hull layer
        str(chullfilled_layer)      # Filled convex hull layer
    ]

    # Execute the command using subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Show the output of the R command
        print("R command output:")
        print(result.stdout)

        # Show any errors if they occur
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # If 'show' is True, attempt to display the PCA plot
        pca_image_path = os.path.join(output_directory, "efourier_results", "pca_plot.png")
        if show and os.path.exists(pca_image_path):
            img = mpimg.imread(pca_image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off the axes
            plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error while executing the R script: {e.stderr}")


def run_obtain_kmeans_classification_r(input_directory, output_directory, img_width=750, img_height=500, 
                                       kmeans_objects_path="", PC_axis1=1, PC_axis2=2, 
                                       chull_layer="FALSE", chullfilled_layer="FALSE", show=True):
    """
    Executes the R script "Obtain_kmeans_classification.R" with the provided arguments.

    Parameters:
    - input_directory (str): Path to the directory containing the PCA object.
    - output_directory (str): Path to the output directory where results will be saved.
    - img_width (int): Width of the image for the clustering plot.
    - img_height (int): Height of the image for the clustering plot.
    - kmeans_objects_path (str): Path to the RDS file with the k-means clustering object.
    - PC_axis1 (int): Principal PCA axis for the plot (default is 1).
    - PC_axis2 (int): Secondary PCA axis for the plot (default is 2).
    - chull_layer (str): If "TRUE", adds a convex hull layer.
    - chullfilled_layer (str): If "TRUE", adds a filled convex hull layer.
    - show (bool): If True, displays the generated clustering plot using matplotlib.
    """

    # Check if the output directory exists, create it if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Path to the R script
    script_r_path = "Obtain_kmeans_classification.R"

    # Create the command to execute the R script with the provided arguments
    command = [
        'Rscript', 
        script_r_path, 
        str(input_directory),      # Path to the directory with the PCA object
        str(output_directory),      # Output directory
        str(img_width),             # Image width
        str(img_height),            # Image height
        str(kmeans_objects_path),   # Path to the k-means RDS file
        str(PC_axis1),              # PC1 axis
        str(PC_axis2),              # PC2 axis
        str(chull_layer),           # Convex hull layer
        str(chullfilled_layer)      # Filled convex hull layer
    ]

    # Execute the command with subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Show the output from the R command
        print("R command output:")
        print(result.stdout)

        # Show any errors if they occur
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # If 'show' is True, attempt to display the clustering plot
        clustered_image_path = os.path.join(output_directory, "kmeans_results", "pca_plot_clustered.png")
        if show and os.path.exists(clustered_image_path):
            img = mpimg.imread(clustered_image_path)
            plt.imshow(img)
            plt.axis('off')  # Disable axes
            plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error executing the R script: {e.stderr}")


def run_kmeans_efourier_r(pca_objects_path, output_directory, max_clusters, img_width_pca=1000, img_height_pca=1000,
                          plot_xlim=250, plot_ylim=250, show=True):
    """
    Executes the R script "kmeans_Efourier_morphometric.R" with the provided arguments.

    Parameters:
    - pca_objects_path (str): Path to the RDS file containing the PCA object.
    - output_directory (str): Directory where the results will be saved.
    - max_clusters (int): Maximum number of clusters to use in k-means.
    - img_width_pca (int): Width of the PCA image.
    - img_height_pca (int): Height of the PCA image.
    - plot_xlim (int): X-axis limit for the plot.
    - plot_ylim (int): Y-axis limit for the plot.
    - show (bool): If True, displays the generated plot with matplotlib.
    """

    # Verify that the output directory exists, create it if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Path to the R script
    script_r_path = "kmeans_efourier_morphometrics.R"
    
    # Create the command to execute the R script with the provided arguments
    command = [
        'Rscript',
        script_r_path,
        str(pca_objects_path),      # Path to the PCA object RDS file
        str(output_directory),      # Output directory
        str(img_width_pca),         # PCA image width
        str(img_height_pca),        # PCA image height
        str(max_clusters),          # Maximum number of clusters
        str(plot_xlim),             # X-axis limit
        str(plot_ylim),             # Y-axis limit
    ]

    # Execute the command with subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Show the output from the R command
        print("R command output:")
        print(result.stdout)

        # Show any errors if they occur
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # If 'show' is True, attempt to display the generated plots
        
        # Final mosaic image name
        mosaic_image_name = os.path.join(output_directory, "kmeans_results", 'kmeans_shape_plot.jpg')
        folder_path = os.path.join(output_directory, "kmeans_results")
        
        # Get and sort images by their names
        images = sorted(os.listdir(folder_path))

        # Load all images into a dictionary organized by cluster number (k)
        images_dict = {}

        # Regular expression to check the file name format
        pattern = r"centroids_k(\d+)_cluster_(\d+)\.jpg"

        for image in images:
            if image.endswith('.jpg'):
                # Use the regular expression to extract the cluster number k and index y
                match = re.match(pattern, image)
                if match:
                    k = int(match.group(1))  # Get the number of clusters (k)
                    y = int(match.group(2))  # Get the cluster index (y)
                    
                    # Add the image to the dictionary
                    if k not in images_dict:
                        images_dict[k] = []
                    images_dict[k].append((y, os.path.join(folder_path, image)))
                else:
                    print(f"Warning: file {image} does not follow the expected format.")
                    continue

        # Sort the clusters within each k
        for k in images_dict:
            images_dict[k].sort()  # Sort images by index y within each k

        # Load the images and calculate the mosaic dimensions
        loaded_images = {k: [Image.open(image[1]) for image in images_dict[k]] for k in images_dict}
        width, height = loaded_images[1][0].size  # Assume all images have the same size
        total_height = sum([height for k in loaded_images])  # Total height of the mosaic
        max_width = max([width * len(loaded_images[k]) for k in loaded_images])  # Maximum width of the mosaic

        # Create a blank image for the mosaic
        mosaic = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        # Place the images in the mosaic
        y_offset = 0
        for k in sorted(loaded_images):
            x_offset = 0
            for img in loaded_images[k]:
                mosaic.paste(img, (x_offset, y_offset))
                x_offset += width
            y_offset += height

        # Save the mosaic image
        mosaic.save(mosaic_image_name)
        
        # Remove the used images
        for k in images_dict:
            for _, image_path in images_dict[k]:
                os.remove(image_path)  # Delete each image used in the mosaic
                print(f"Deleted: {image_path}")

        # If 'show' is True and the mosaic exists, display it
        if show and os.path.exists(mosaic_image_name):
            img = mpimg.imread(mosaic_image_name)
            plt.imshow(img)
            plt.axis('off')  # Disable axes
            plt.show()

        # Check and display the elbow method plot if available
        elbow_image_path = os.path.join(output_directory, "kmeans_results", "Elbow_method_plot.jpg")
        if show and os.path.exists(elbow_image_path):
            img = mpimg.imread(elbow_image_path)
            plt.imshow(img)
            plt.axis('off')  # Disable axes
            plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error executing the R script: {e.stderr}")


def process_images_and_perform_pca(directory, working_directory, n_components=50, k_max=10, std_multiplier=2):
    """
    This function processes binary images in a specified directory, performs Principal Component Analysis (PCA) on the images, 
    and evaluates different KMeans clustering solutions. It also generates visualizations for the PCA components and clusters.
    
    Parameters:
    - directory (str): The path to the directory containing the input image files (e.g., '.png', '.jpg').
    - working_directory (str): The directory where the output files will be saved.
    - n_components (int, default=50): The number of principal components to retain in the PCA. 
    - k_max (int, default=10): The maximum number of clusters to evaluate for KMeans clustering.
    - std_multiplier (float, default=2): A multiplier used to adjust the standard deviation of the principal components for visualization. 
      It controls how much variance from the mean shape is visualized.
    
    Returns:
    - Saves several output files:
      - 'pca_values.txt': The PCA values for each image (first 10 principal components).
      - 'explained_variance.txt': The explained variance ratio for each principal component.
      - A series of images visualizing the influence of each principal component.
      - KMeans clustering centroid images for different values of k.
      - A combined "staircase" image of centroids.
      - An elbow plot showing the Within-Group Sum of Squares (WGSS) for different k values.
    """
    # Step 1: Load images and convert them to binary arrays
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
    # Create a custom color map from white to light brown
    colors = ["white", "#f5b041"]  # White to Light Brown
    cmap_brown = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    images = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).convert('1')  # Convert to binary image
        image_array = np.array(image)  # Convert the image to a numpy array
        image_array = np.invert(image_array)  # Invert the binary image
        images.append(image_array)
    
    # Step 2: Flatten the list of images into a matrix of shape (k, m*n), where k is the number of images
    images = np.array(images)
    flattened_images = images.reshape(images.shape[0], -1)  # Flatten the images into vectors
    print("Flattened image matrix shape:", flattened_images.shape)

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_images = pca.fit_transform(flattened_images)

    # Explained variance for each component
    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame with the PCA values
    df_pca = pd.DataFrame(
        pca_images[:, :10],  # Take only the first 10 principal components
        columns=[f"PC{i+1}" for i in range(10)],  # Column names (PC1 to PC10)
        index=image_files  # Use actual image filenames as the index
    )

    # Save the PCA values DataFrame as a TXT file
    pca_output_file = os.path.join(working_directory, "pca_values.txt")
    df_pca.to_csv(pca_output_file, sep="\t", index=True)
    print(f"PCA values file saved as {pca_output_file}")

    # Save the explained variance in another TXT file
    variance_output_file = os.path.join(working_directory, "explained_variance.txt")
    with open(variance_output_file, "w") as f:
        f.write("Principal Component\tExplained Variance\n")
        for i, var in enumerate(explained_variance, 1):
            f.write(f"PC{i}\t{var:.6f}\n")
    
    print(f"Explained variance file saved as {variance_output_file}")

    # Step 3: Calculate the mean shape in the original space
    mean_shape = pca.mean_.reshape(images.shape[1], images.shape[2])

    # For each principal component (PC1 to PC10)
    for pc in range(10):  # Iterate over the first 10 PCs
        std_pc = np.sqrt(pca.explained_variance_[pc])  # Standard deviation of the component
        direction_pc = pca.components_[pc].reshape(images.shape[1], images.shape[2])  # Direction of the PC in the original space

        # Calculate the adjusted shapes based on mean shape and standard deviation
        shape_pos = mean_shape + std_multiplier * std_pc * direction_pc  # Mean + std_multiplier * std
        shape_neg = mean_shape - std_multiplier * std_pc * direction_pc  # Mean - std_multiplier * std

        # Create a figure with 3 images: (-std_multiplier * std, mean, +std_multiplier * std)
        plt.figure(figsize=(12, 4))

        # Mean shape - std_multiplier * std
        plt.subplot(1, 3, 1)
        plt.imshow(shape_neg > 0.5, cmap=cmap_brown)  # Threshold to binarize the image
        plt.title(f"PC{pc+1}: Mean - {std_multiplier}*std")
        plt.axis("off")

        # Mean shape
        plt.subplot(1, 3, 2)
        plt.imshow(mean_shape > 0.5, cmap=cmap_brown)  # Threshold to binarize the image
        plt.title(f"PC{pc+1}: Mean")
        plt.axis("off")

        # Mean shape + std_multiplier * std
        plt.subplot(1, 3, 3)
        plt.imshow(shape_pos > 0.5, cmap=cmap_brown)  # Threshold to binarize the image
        plt.title(f"PC{pc+1}: Mean + {std_multiplier}*std")
        plt.axis("off")

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(working_directory, f"pc{pc+1}_influence.jpg"), format="jpg")
        plt.show()

    # List to store the paths of images generated for each k
    cluster_images = []

    # Step 4: Evaluate KMeans for different values of k (1 to k_max)
    wgss = []  # To store WGSS (Within-Group Sum of Squares) for each k
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_images)

        # Calculate WGSS (Within-Group Sum of Squares)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        wgss_value = sum(
            np.sum((pca_images[labels == cluster] - cluster_centers[cluster]) ** 2)
            for cluster in range(k)
        )
        wgss.append(wgss_value)

        # Visualize the centroids of each cluster
        centroids_pca = kmeans.cluster_centers_

        # Project centroids back to the original space
        centroids_original = pca.inverse_transform(centroids_pca)

        # Visualize centroids in a single row
        plt.figure(figsize=(3 * k, 3))  # Adjust figure size to accommodate k images in a single row
        for i, centroid in enumerate(centroids_original):
            # Convert the centroid (vector) to a binary image
            binary_image = centroid.reshape(images.shape[1], images.shape[2]) > 0.5  # Binarize with threshold

            # Create a subplot
            ax = plt.subplot(1, k, i + 1)
            ax.imshow(binary_image, cmap=cmap_brown)  # Use a brown colormap
            ax.set_title(f'Centroid {i + 1}')
            ax.axis('off')

        # Save centroid image as JPG
        output_path = os.path.join(working_directory, f"centroids_k_{k}.jpg")
        cluster_images.append(output_path)  # Add the image path to the list
        plt.tight_layout()
        plt.savefig(output_path, format="jpg")
        plt.show()

    # Step 6: Combine all cluster images into one "staircase" image
    combined_height = 0
    max_width = 0
    images_to_combine = []

    for image_path in cluster_images:
        img = Image.open(image_path)
        images_to_combine.append(img)
        combined_height += img.height
        max_width = max(max_width, img.width)

    # Create a new blank image with the total height and max width
    staircase_image = Image.new('RGB', (max_width, combined_height), (255, 255, 255))

    # Paste each image one below the other
    y_offset = 0
    for img in images_to_combine:
        staircase_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the final combined "staircase" image
    staircase_image_path = os.path.join(working_directory, "centroids_staircase.jpg")
    staircase_image.save(staircase_image_path)
    print(f"Combined staircase image saved as {staircase_image_path}")

    # Step 7: Evaluate optimal number of clusters using the WGSS Elbow method
    plt.plot(range(1, k_max + 1), wgss, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WGSS (Within-Group Sum of Squares)')
    plt.title('Elbow Method for Selecting k (WGSS)')
    plt.savefig(os.path.join(working_directory, "elbow_plot_wgss.jpg"), format="jpg")
    plt.show()