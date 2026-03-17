# Morphometrics install packages

# Specify the CRAN repository
repos <- "https://cloud.r-project.org/"

# Check if the "Momocs" package is installed
if (!requireNamespace("Momocs", quietly = TRUE)) {
  cat("The 'Momocs' package is not installed. Installing...\n")
  install.packages("Momocs", repos = repos)
} else {
  cat("The 'Momocs' package is already installed.\n")
}

# Check if the "dplyr" package is installed
if (!requireNamespace("dplyr", quietly = TRUE)) {
  cat("The 'dplyr' package is not installed. Installing...\n")
  install.packages("dplyr", repos = repos)
} else {
  cat("The 'dplyr' package is already installed.\n")
}
