library(Momocs)

# Load the PCA object from the .rds file
args <- commandArgs(trailingOnly = TRUE)

# Check if sufficient arguments are passed
if (length(args) >= 7) {
  pca_objects_path <- args[1]  # Path to the PCA objects file
  output_directory <- args[2]  # Output directory
  img_width <- as.numeric(args[3])    # Image width for the panel
  img_height <- as.numeric(args[4])   # Image height for the panel
  max_clusters <- as.numeric(args[5]) # Maximum number of clusters for k-means
  plot_xlim <- as.numeric(args[6])   # X-axis limits for the plot
  plot_ylim <- as.numeric(args[7])   # Y-axis limits for the plot
} else {
  stop("Not enough arguments were passed.")
}

output_folder <- file.path(output_directory, "kmeans_results")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

pca_fourier <- readRDS(pca_objects_path)

nb_h <- as.numeric(length(pca_fourier$eig) / 4)

# Initialize the list to store the reconstructed shapes of the centroids
centroid_shapes <- list()
wss_values <- numeric(max_clusters)

# Loop to perform k-means for each number of clusters from 1 to the maximum
for (num_clusters in 1:max_clusters) {
  
  # Perform k-means with the current number of clusters
  kmeans_result <- kmeans(pca_fourier$x, centers=num_clusters)
  saveRDS(kmeans_result, file = file.path(output_folder, paste0("kmeans_pca_fourier_", num_clusters, ".rds")))
  wss_values[num_clusters] <- kmeans_result$tot.withinss  # Within-group sum of squares
  centroids_pca <- kmeans_result$centers
  
  # Initialize the list to store the reconstructed shapes of the centroids
  centroid_shapes <- list()
  
  # Project each centroid into the Fourier space
  for (i in 1:num_clusters) {
    # Centroids in the PCA space
    projections <- as.matrix(centroids_pca[i, , drop = FALSE])  # Convert to matrix
    projections <- t(projections)
    
    # Project to the Fourier space and add the mean (mshape)
    reconstructed_coef <- as.vector(pca_fourier$rotation %*% projections) + pca_fourier$mshape
    
    # Number of harmonics (adjust according to your case)
    # Change this value if you used a different number of harmonics
    
    # Split the reconstructed coefficients into an, bn, cn, dn
    an <- reconstructed_coef[1:nb_h]
    bn <- reconstructed_coef[(nb_h + 1):(2 * nb_h)]
    cn <- reconstructed_coef[(2 * nb_h + 1):(3 * nb_h)]
    dn <- reconstructed_coef[(3 * nb_h + 1):(4 * nb_h)]
    
    # Use the original center of the shape to maintain displacement
    ao <- pca_fourier$center[1]
    co <- pca_fourier$center[2]
    
    # Create an efourier object with the reconstructed coefficients
    ef_centroid <- list(an = an, bn = bn, cn = cn, dn = dn, ao = ao, co = co)
    
    # Reconstruct the shape and store it in the list
    centroid_shapes[[i]] <- efourier_i(ef_centroid)
  }
  
  # Visualize the reconstructed centroid shapes and save the image
  for (i in 1:num_clusters) {
    # Rotate the figure coordinates 90 degrees counterclockwise (pi/2 radians)
    centroid_shapes[[i]] <- coo_rotate(centroid_shapes[[i]], theta = pi / 2)
    
    # Create the filename to save the image
    output_file <- paste0("centroids_k", num_clusters, "_cluster_", i, ".jpg")
    
    # Save each image as a .jpg file
    jpeg(filename = file.path(output_folder, output_file), width = img_width, height = img_height)
    coo_plot(centroid_shapes[[i]], border='orange3', col = 'orange3',
             xy.axis = FALSE, xlim = c(-plot_xlim, plot_xlim), ylim = c(-plot_ylim, plot_ylim))  # Show each reconstructed centroid
    text(0, 0, paste("k =", i), col="white", cex=1)  # Add the label "k=1", "k=2", etc.
    dev.off()
    cat("Centroid image with", num_clusters, "clusters saved as", output_file, "\n")
  }
}

# Elbow method
output_file <- paste0("Elbow_method_plot.jpg")
jpeg(filename = file.path(output_folder, output_file), width = img_width, height = img_height)
plot(1:max_clusters, wss_values, type="b", pch=19, col="blue", 
     xlab="Number of clusters (k)", ylab="WGSS (Within-Group Sum of Squares)",
     main="Elbow Method for Selecting k (WGSS)")
dev.off()
