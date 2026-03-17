library(Momocs)

# Load the PCA object from the .rds file
args <- commandArgs(trailingOnly = TRUE)

# Check if sufficient arguments are passed
if (length(args) >= 7) {
  pca_objects_path <- args[1]   # Path to the PCA objects file
  output_directory <- args[2]   # Output directory
  img_width_pca <- as.numeric(args[3])    # Image width for the panel
  img_height_pca <- as.numeric(args[4])   # Image height for the panel
  grouping_factor <- args[5]    # Grouping factor for plotting
  PC_axis1 <- as.numeric(args[6])   # Principal component axis 1
  PC_axis2 <- as.numeric(args[7])   # Principal component axis 2
} else {
  stop("Not enough arguments were passed.")
}

output_folder <- file.path(output_directory, "efourier_results")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

pca_fourier <- readRDS(pca_objects_path)

# Save the PCA plot as a PNG image
png(filename = file.path(output_folder, "pca_plot.png"), 
    width = img_width_pca, 
    height = img_height_pca)

if (nzchar(grouping_factor) > 0) {
  # If grouping_factor is provided, convert it to a factor and plot the PCA
  grouping_factor <- as.factor(grouping_factor)
  plot_PCA(pca_fourier, f = grouping_factor, axes = c(PC_axis1, PC_axis2))
} else {
  # Otherwise, plot the PCA without grouping
  plot_PCA(pca_fourier, axes = c(PC_axis1, PC_axis2), points = TRUE)
}

dev.off()