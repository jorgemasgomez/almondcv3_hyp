# E_fourier_morphometrics
library(Momocs)

# Load the PCA object from the .rds file
args <- commandArgs(trailingOnly = TRUE)

# Check if enough arguments are passed
if (length(args) >= 10) {
  outline_objects_path <- args[1]  # Path to the outline objects
  nharmonics <- as.numeric(args[2])  # Number of harmonics
  output_directory <- args[3]  # Output directory
  img_width_boxplot <- as.numeric(args[4])  # Image width for the boxplot panel
  img_height_boxplot <- as.numeric(args[5])  # Image height for the boxplot panel
  img_width_pca <- as.numeric(args[6])  # Image width for the PCA panel
  img_height_pca <- as.numeric(args[7])  # Image height for the PCA panel
  normalize <- as.logical(args[8])  # Whether to normalize or not
  start_point <- as.logical(args[9])  # Whether to use a starting point for the Fourier series
  align_x <- as.logical(args[10])  # Whether to align along the X-axis
} else {
  stop("Not enough arguments provided.")
}

output_folder <- file.path(output_directory, "efourier_results")

# Create output folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# Read the outline objects from the file
outline_objects <- readRDS(outline_objects_path)

# Align outlines if required
if (align_x) {
  outline_objects <- coo_slidedirection(outline_objects, direction = "right", center = TRUE)
  outline_objects <- coo_alignxax(outline_objects)
}

# Perform Fourier Transform on the outline objects
e_fourier_contours <- efourier(outline_objects, nb.h = nharmonics, norm = normalize, start = start_point)

# Save the Fourier coefficients to a file
write.table(e_fourier_contours$coe, 
            file = file.path(output_folder, "e_fourier_coefs.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)

# Create and save a boxplot as a PNG image
png(filename = file.path(output_folder, "boxplot_output.png"), 
    width = img_width_boxplot, 
    height = img_height_boxplot)
boxplot(e_fourier_contours, drop = 1)
dev.off()

# Perform PCA on the Fourier results
pca_fourier <- PCA(e_fourier_contours)

# Save the PCA coordinates to a text file
write.table(pca_fourier$x, 
            file = file.path(output_folder, "e_fourier_pcs_coordinates.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)

# Save the PCA eigenvalues to a text file
write.table(pca_fourier$eig, 
            file = file.path(output_folder, "e_fourier_pcs_eigenvalues.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)

# Create and save a PCA plot as a PNG image
png(filename = file.path(output_folder, "pca_output.png"), 
    width = img_width_pca, 
    height = img_height_pca)
plot_PCA(pca_fourier)
dev.off()

# Save the PCA object as an .rds file
saveRDS(pca_fourier, file = file.path(output_folder, "pca_fourier.rds"))