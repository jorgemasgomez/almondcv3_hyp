# Load necessary packages
library(Momocs)
library(dplyr)

# Capture the arguments passed to the script
args <- commandArgs(trailingOnly = TRUE)

# Check if enough arguments are provided
if (length(args) >= 16) {
  info_data <- args[1]               # File with the data
  grouping_factor <- args[2]         # Grouping factor
  directory <- args[3]               # Directory where the .jpg files are located
  img_width_panel <- as.numeric(args[4])    # Width of the image for the panel
  img_height_panel <- as.numeric(args[5])   # Height of the image for the panel
  img_width_stack <- as.numeric(args[6])    # Width of the image for the stack
  img_height_stack <- as.numeric(args[7])   # Height of the image for the stack
  output_directory <- args[8]        # Output directory for images
  nexamples <- as.numeric(args[9])   # Number of examples for the loop
  nharmonics <- as.numeric(args[10]) # Number of harmonics for the functions to calibrate
  img_width_ptolemy <- as.numeric(args[11])    # Width of the image for Ptolemy
  img_height_ptolemy <- as.numeric(args[12])   # Height of the image for Ptolemy
  img_width_deviations <- as.numeric(args[13]) # Width of the image for calibrate_deviations_efourier
  img_height_deviations <- as.numeric(args[14])# Height of the image for calibrate_deviations_efourier
  img_width_reconstructions <- as.numeric(args[15]) # Width of the image for calibrate_reconstructions_efourier
  img_height_reconstructions <- as.numeric(args[16])# Height of the image for calibrate_reconstructions_efourier
} else {
  stop("Not enough arguments were provided.")
}

# Check if the info_data file exists and is not empty
if (nzchar(info_data) && file.exists(info_data) && file.info(info_data)$size > 0) {
  # If the file exists and is not empty, load it
  info_data_df <- read.table(info_data, sep = "\t", header = TRUE)
  info_data_df <- info_data_df %>% mutate_all(as.factor)
} else {
  # If info_data is invalid, remove the variable
  cat("The info_data file is not valid or is empty. The use of this variable will be skipped.\n")
  rm(info_data)  # Remove the info_data variable
}

# Create the exploratory_plots folder inside the output_directory if it doesn't exist
output_folder <- file.path(output_directory, "exploratory_plots")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# Create the vector with the .jpg file paths in the directory
jpg_path_list <- list.files(path = directory, pattern = "\\.jpg$", full.names = TRUE)
outlines_objects <- import_jpg(jpg.paths = jpg_path_list)

# Set outlines_objects with fac if info_data_df exists
if (exists("info_data_df")) {
  outlines_objects <- Out(x = outlines_objects, fac = info_data_df)
  
  # Save the panel image
  png(filename = file.path(output_folder, "panel_output.png"), width = img_width_panel, height = img_height_panel)
  panel(outlines_objects, fac = info_data_df[[grouping_factor]])
  dev.off()
  
  # Save the stack image
  png(filename = file.path(output_folder, "stack_output.png"), width = img_width_stack, height = img_height_stack)
  stack(outlines_objects)
  dev.off()
  
} else {
  outlines_objects <- Out(x = outlines_objects)
  
  # Save the panel image
  png(filename = file.path(output_folder, "panel_output.png"), width = img_width_panel, height = img_height_panel)
  panel(outlines_objects)
  dev.off()
  
  # Save the stack image
  png(filename = file.path(output_folder, "stack_output.png"), width = img_width_stack, height = img_height_stack)
  stack(outlines_objects)
  dev.off()
}

# Save the outlines_objects as an .rds file
saveRDS(outlines_objects, file = file.path(output_folder, "outlines_objects.rds"))

# Loop to run Ptolemy, calibrate_deviations_efourier, and calibrate_reconstructions_efourier
for (i in 1:nexamples) {
  # Select a random index within the outlines_objects
  random_index <- sample(1:length(outlines_objects), 1)
  
  # Save the Ptolemy image with custom size
  png(filename = file.path(output_folder, paste0("ptolemy_output_", i, ".png")), width = img_width_ptolemy, height = img_height_ptolemy)
  Ptolemy(outlines_objects[random_index], nb.h = nharmonics)
  dev.off()
  
  # Save the calibrate_deviations_efourier image with custom size
  png(filename = file.path(output_folder, paste0("deviations_efourier_output_", i, ".png")), width = img_width_deviations, height = img_height_deviations)
  calibrate_deviations_efourier(outlines_objects, range = 1:nharmonics, plot = TRUE)
  dev.off()
  
  # Save the calibrate_reconstructions_efourier image with custom size
  png(filename = file.path(output_folder, paste0("reconstructions_efourier_output_", i, ".png")), width = img_width_reconstructions, height = img_height_reconstructions)
  print(calibrate_reconstructions_efourier(outlines_objects, range = c(1:nharmonics)))
  dev.off()
}