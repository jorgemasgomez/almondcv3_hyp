"""
exploration_modelling_functions.py

This module contains functions for exploratory analysis and modelling of spectral data.
It provides tools to:

- Perform dimensionality reduction (e.g., PCA) for spectral datasets.
- Visualize spectra as line plots for individual samples or groups.
- Build regression models such as PLS and iPLS for prediction of target variables.
- Apply and explore preprocessing treatments such as COW (Correlation Optimized Warping).
- Evaluate model performance and visualize results with standard metrics.

The functions are designed to work with hyperspectral or multispectral datasets,
typically organized as arrays or dataframes with samples, wavelengths..

Dependencies include: numpy, pandas, matplotlib, seaborn, sklearn, scipy, joblib, astartes.
"""

import os
import math
import ast
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
import astartes as at  # For Kennard-Stone


def plot_preprocessing_results(df_all_results, unique_ids, preprocessing, results_directory="results",
                               ids="ID", plot_type="individual", color_col=None, mean_centering=False,
                               xlim=(900, 1748), ylim=None, vline=None):
    """
    Generates and saves plots for selected preprocessing metrics of spectral data.

    Parameters
    ----------
    df_all_results : pandas.DataFrame
        DataFrame containing the spectral data and preprocessing metrics.
    unique_ids : list
        List of unique sample IDs to process.
    preprocessing : list
        List of preprocessing metric column names to plot (e.g., ['Mean_SNV', 'Mean_MSC']).
    results_directory : str, optional
        Folder where the generated plots will be saved. Default is 'results'.
    ids : str, optional
        Name of the column in df_all_results that contains the sample IDs. Default is 'ID'.
    plot_type : str, optional
        Type of plot to generate: "individual", "combined", or "both". Default is "individual".
    color_col : str, optional
        Column name used to define line colors. Can be numeric or categorical. Default is None.
    mean_centering : bool, optional
        If True, the mean per band and preprocessing metric is subtracted before plotting. Default is False.
    xlim : tuple, optional
        X-axis limits (default 900–1748 nm).
    ylim : tuple, optional
        Y-axis limits. If None, axis is auto-scaled.
    vline : float, optional
        X-axis position for a vertical reference line. Default is None.

    Notes
    -----
    - Individual plots show each sample separately, with optional shading for standard deviation if available.
    - Combined plots overlay all samples for a given metric, optionally colored by a feature.
    - The function creates the results directory if it does not exist.
    """
    
    # Create results directory if it does not exist
    os.makedirs(results_directory, exist_ok=True)

    # Apply mean centering per band and metric if requested
    if mean_centering:
        df_all_results_processed = df_all_results.copy()
        for method in preprocessing:
            df_all_results_processed[method] = df_all_results[method] - \
                df_all_results.groupby("Band")[method].transform("mean")
    else:
        df_all_results_processed = df_all_results.copy()

    # ----- Individual plots -----
    if plot_type in ["individual", "both"]:
        for id_ in unique_ids:
            df_filtered = df_all_results_processed[df_all_results_processed[ids] == id_]

            num_metrics = len(preprocessing)
            num_cols = math.ceil(math.sqrt(num_metrics))
            num_rows = math.ceil(num_metrics / num_cols)

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
            fig.suptitle(f'Spectra - ID: {id_}')
            axs = np.array(axs).flatten()

            for i, metric in enumerate(preprocessing):
                if metric in df_filtered.columns:
                    axs[i].plot(df_filtered['Band'], df_filtered[metric])

                    # Add shading for standard deviation if available
                    std_metric = metric.replace("Mean", "Std").replace("Median", "Std")
                    if std_metric in df_filtered.columns:
                        axs[i].fill_between(df_filtered['Band'], 
                                            df_filtered[metric] - df_filtered[std_metric],
                                            df_filtered[metric] + df_filtered[std_metric],
                                            alpha=0.2)

                    axs[i].set_title(metric)
                    axs[i].set_xlabel('Bands (nm)')
                    axs[i].set_ylabel('Reflectance')

            # Remove empty axes
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout()
            plt.savefig(f'{results_directory}/Spectra_{id_}.png')
            plt.show()

    # ----- Combined plots -----
    if plot_type in ["combined", "both"]:
        for metric in preprocessing:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.title(f'Combined Spectra - {metric}')
            plt.xlabel('Bands (nm)')
            plt.ylabel('Reflectance')

            # Define color mapping
            if color_col:
                if pd.api.types.is_numeric_dtype(df_all_results_processed[color_col]):
                    norm = plt.Normalize(vmin=df_all_results_processed[color_col].min(),
                                         vmax=df_all_results_processed[color_col].max())
                    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                    color_map = sm
                else:
                    unique_groups = df_all_results_processed[color_col].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
                    group_color_map = {group: color for group, color in zip(unique_groups, colors)}
            else:
                color_map = 'black'

            # Plot all IDs
            for id_ in unique_ids:
                df_filtered = df_all_results_processed[df_all_results_processed[ids] == id_]

                if color_col:
                    if pd.api.types.is_numeric_dtype(df_filtered[color_col]):
                        color = color_map.to_rgba(df_filtered[color_col].iloc[0])
                    else:
                        group = df_filtered[color_col].iloc[0]
                        color = group_color_map.get(group, "black")
                else:
                    color = 'black'

                ax.plot(df_filtered['Band'], df_filtered[metric], color=color)

            # Add colorbar for numeric color columns
            if color_col and pd.api.types.is_numeric_dtype(df_all_results_processed[color_col]):
                cbar = fig.colorbar(color_map, ax=ax)
                cbar.set_label(color_col)

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            else:
                ax.autoscale(enable=True, axis='y', tight=True)
            if vline is not None:
                ax.axvline(x=vline, color='red', linestyle='--', linewidth=1)

            plt.tight_layout()
            plt.savefig(f'{results_directory}/Combined_Spectra_{metric}_{"mean_centered" if mean_centering else "raw"}.png')
            plt.show()

def plot_mean_std_spectrum(df_all_results, preprocessing_metric, results_directory="results", xlim=(900, 1748), ylim=None):
    """
    Plots the mean spectrum and standard deviation for all samples.

    Parameters
    ----------
    df_all_results : pandas.DataFrame
        DataFrame containing columns 'Band' and the selected preprocessing metric.
    preprocessing_metric : str
        Name of the preprocessing metric to plot (e.g., 'Mean_SpectralValue').
    results_directory : str, optional
        Folder where the generated plot will be saved. Default is 'results'.
    xlim : tuple, optional
        X-axis limits (default 900–1748 nm).
    ylim : tuple, optional
        Y-axis limits. If None, axis is auto-scaled.
    """
    # Create results directory if it does not exist
    os.makedirs(results_directory, exist_ok=True)

    # Group by wavelength ('Band') and calculate mean and standard deviation
    grouped = df_all_results.groupby('Band')[preprocessing_metric]
    mean_vals = grouped.mean()
    std_vals = grouped.std()

    bands = mean_vals.index
    mean = mean_vals.values
    std = std_vals.values

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(bands, mean, color='blue', label='Mean Spectrum')
    plt.fill_between(bands, mean - std, mean + std, color='lightblue', alpha=0.5, label='Std Dev')

    # Labels and style
    plt.xlabel('Bands (nm)', fontsize=18)
    plt.ylabel('Reflectance', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    # Axis limits
    plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()

    # Save high-resolution figure
    output_path = os.path.join(results_directory, f'Mean_STD_Spectrum_{preprocessing_metric}.png')
    plt.savefig(output_path, dpi=300)

    plt.show()

def pca_preprocessing(df_long, preprocessing_metric, unique_sample_variable, color_variable=None, 
                      n_components=2, show_labels=False, label_size=8, components_to_plot=(1, 2), 
                      hotelling=False, std_scaling=True, number_variable_loading=20, title="", 
                      legendx=0.95, legendy=0.95, results_path="path", legend_font_size=8):
    """
    Performs PCA on a selected spectral metric from a long-format DataFrame and visualizes results.

    Key functionalities:
    - Reduces dimensionality of spectral data (samples x bands).
    - Calculates Q residuals for each sample to detect deviations from the PCA model.
    - Optionally calculates Hotelling's T² for multivariate outlier detection.
    - Visualizes the top influential variables in component space (based on loadings).
    - Generates PCA scatter plots colored by a metadata variable.
    - Plots cumulative explained variance for the computed components.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format DataFrame with columns ['Metric', 'Value', sample metadata...].
    preprocessing_metric : str
        Spectral metric/column to use for PCA (e.g., 'Mean_SG1_W15_P2').
    unique_sample_variable : str
        Column identifying individual samples (e.g., 'ID').
    color_variable : str, optional
        Column used to color samples in PCA plot (numeric or categorical).
    n_components : int, optional
        Number of principal components to compute (default 2).
    show_labels : bool, optional
        Whether to display sample labels on PCA scatter plot.
    label_size : int, optional
        Font size for sample labels.
    components_to_plot : tuple, optional
        PCs to plot (default (1, 2)).
    hotelling : bool, optional
        If True, calculates Hotelling's T² and marks multivariate outliers.
    std_scaling : bool, optional
        Whether to standardize spectral values before PCA.
    number_variable_loading : int, optional
        Number of top influential variables to highlight in loadings scatter plot.
    title : str, optional
        Optional plot title.
    legendx, legendy : float, optional
        Coordinates for legend placement.
    results_path : str, optional
        Folder to save PCA plots.
    legend_font_size : int, optional
        Font size for legend text.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object containing components, explained variance, etc.
    pca_df : pandas.DataFrame
        DataFrame with PCA scores per sample, Q residuals, and Hotelling's T² (if requested).

    Notes
    -----
    - Top influential variables are determined by sum of squares of loadings in selected PCs.
    - Q residuals highlight samples that deviate from the PCA reconstruction.
    - Hotelling's T² detects multivariate outliers in the PCA space.
    - The function produces multiple plots: PCA scatter, loadings, Hotelling vs Q residuals, and explained variance.
    """

    # Filter DataFrame for the selected preprocessing metric
    df_metric = df_long[df_long['Metric'] == preprocessing_metric]

    # Pivot data: samples as rows, bands as columns
    df_pivot = df_metric.pivot_table(index=unique_sample_variable, columns='Band', values='Value', aggfunc='first')
    df_pivot = df_pivot.dropna()

    # Standardize/scale the data
    scaler = StandardScaler(with_std=std_scaling)
    df_scaled = scaler.fit_transform(df_pivot)

    # PCA computation
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled)

    # Create DataFrame with PCA scores
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df[unique_sample_variable] = df_pivot.index

    # --------------------------
    # Identify top influential variables for selected PCs
    # --------------------------
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=df_pivot.columns)

    pc_x = f'PC{components_to_plot[0]}'
    pc_y = f'PC{components_to_plot[1]}'

    # Importance = sum of squares of loadings in selected PCs
    loadings['importance'] = loadings[pc_x]**2 + loadings[pc_y]**2

    # Select top influential variables
    top_variables = loadings.nlargest(number_variable_loading, 'importance')

    # Scatter plot of top variables
    plt.figure(figsize=(10, 8))
    plt.scatter(top_variables[pc_x], top_variables[pc_y], color='darkorange', alpha=0.9)
    for i, band in enumerate(top_variables.index):
        plt.text(top_variables[pc_x].iloc[i], top_variables[pc_y].iloc[i], str(band), fontsize=9)

    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.xlabel(f'Loadings on {pc_x}')
    plt.ylabel(f'Loadings on {pc_y}')
    plt.title(f'Top {number_variable_loading} Influential Variables on {pc_x} vs {pc_y}')
    plt.grid(True)
    plt.show()

    # Compute Q residuals (reconstruction error)
    reconstructed = pca.inverse_transform(pca_result)
    q_residuals = np.sum((df_scaled - reconstructed) ** 2, axis=1)
    pca_df['Q_residual'] = q_residuals

    # Detect outliers based on Q residuals
    q_threshold = np.percentile(q_residuals, 95)
    q_outliers = pca_df[pca_df['Q_residual'] > q_threshold]

    if not q_outliers.empty:
        print("Outliers based on Q residuals:")
        print(q_outliers[[unique_sample_variable, 'Q_residual']])
    else:
        print("No outliers detected based on Q residuals.")

    # Calculate Hotelling's T² if requested
    if hotelling:
        pca_mean = np.mean(pca_result, axis=0)
        pca_cov = np.cov(pca_result.T)
        inv_pca_cov = np.linalg.inv(pca_cov)

        t2_values = np.array([
            np.dot(np.dot((x - pca_mean), inv_pca_cov), (x - pca_mean).T) for x in pca_result
        ])
        pca_df['Hotelling_T2'] = t2_values

        critical_value = chi2.ppf(0.95, df=n_components)
        t2_outliers = pca_df[pca_df['Hotelling_T2'] > critical_value]

        if not t2_outliers.empty:
            print("Outliers based on Hotelling's T²:")
            print(t2_outliers[[unique_sample_variable, 'Hotelling_T2']])
        else:
            print("No outliers detected based on Hotelling's T².")

        # Hotelling's T² scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_df[pc_x], pca_df[pc_y], c=t2_values, cmap='coolwarm', alpha=0.7)
        plt.scatter(t2_outliers[pc_x], t2_outliers[pc_y], color='red', label='T² Outliers', alpha=1, edgecolors='black')
        plt.colorbar(scatter, label="Hotelling's T²")
        plt.xlabel(pc_x)
        plt.ylabel(pc_y)
        plt.title(f'PCA - {preprocessing_metric} with Hotelling\'s T²')
        plt.legend()
        plt.show()

        # Plot Hotelling vs Q residuals
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['Hotelling_T2'], pca_df['Q_residual'], alpha=0.7, edgecolors='k')
        plt.axvline(x=critical_value, color='red', linestyle='--', label="T² Threshold (95%)")
        plt.axhline(y=q_threshold, color='orange', linestyle='--', label="Q Threshold (95%)")
        plt.xlabel("Hotelling's T²")
        plt.ylabel("Q residual")
        plt.title(f'Outlier Detection - PCA ({preprocessing_metric})')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Scatter plot colored by Q residuals
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_df[pc_x], pca_df[pc_y], c=q_residuals, cmap='plasma', alpha=0.7)
    plt.scatter(q_outliers[pc_x], q_outliers[pc_y], color='black', label='Q Outliers', alpha=1, edgecolors='white')
    plt.colorbar(scatter, label="Q residual")
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.axis('equal')
    plt.title(f'PCA - {preprocessing_metric} with Q residuals')
    plt.legend()
    plt.show()

    # Scatter plot colored by metadata variable if provided
    plt.figure(figsize=(8, 6))
    if color_variable:
        color_values = df_metric.groupby(unique_sample_variable)[color_variable].first().reindex(pca_df[unique_sample_variable])
        if color_values.dtype == 'O':
            unique_colors = color_values.unique()
            color_map = {val: plt.cm.get_cmap('tab20')(i / len(unique_colors)) for i, val in enumerate(unique_colors)}
            color_list = [color_map[val] for val in color_values]
            scatter = plt.scatter(pca_df[pc_x], pca_df[pc_y], c=color_list, alpha=0.7)
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[val], markersize=8) for val in unique_colors]
            plt.legend(handles, unique_colors, loc='upper left', fontsize=legend_font_size)
        else:
            cmap = cm.viridis
            scatter = plt.scatter(pca_df[pc_x], pca_df[pc_y], c=color_values, cmap=cmap, alpha=0.7)
            plt.colorbar(scatter, label=color_variable)
    else:
        plt.scatter(pca_df[pc_x], pca_df[pc_y], alpha=0.7)

    # Optional labels
    if show_labels:
        num_labels = min(1000, len(pca_df))
        for i, txt in enumerate(pca_df[unique_sample_variable].iloc[:num_labels]):
            plt.text(pca_df[pc_x].iloc[i], pca_df[pc_y].iloc[i], str(txt), fontsize=label_size)

    explained_variance_ratio = pca.explained_variance_ratio_[:n_components]
    plt.xlabel(f'{pc_x} ({explained_variance_ratio[components_to_plot[0]-1]*100:.2f}%)', fontsize=14)
    plt.ylabel(f'{pc_y} ({explained_variance_ratio[components_to_plot[1]-1]*100:.2f}%)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    output_path = os.path.join(results_path, f'PCA.png')
    plt.axis('equal')
    plt.title(title)
    plt.savefig(output_path, dpi=300)
    plt.show()

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_[:n_components]), 
             marker='o', color='b')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Explained Variance of the First {n_components} Components')
    plt.grid(True)
    plt.show()

    return pca, pca_df

def train_and_save_pls_models(
        df_long,
        df_config,
        save_path,
        index_column='ID',
        test_size=0.2,
        scatterplot=True,
        random_state=10,
        scaling=False,
        intervalpls=False      # <-- NUEVO ARGUMENTO
):
    """
    Trains and saves PLS models (and plots) for each row in `df_config`.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format DataFrame with columns ['Metric', 'Value', sample metadata...].
    df_config : pandas.DataFrame
        Configuration DataFrame containing columns ['Trait', 'Metric', 'Number of Components',
        optional 'Bands_selected', and metrics for plotting].
    save_path : str
        Root directory to save models and plots.
    index_column : str, optional
        Column identifying individual samples (default 'ID').
    test_size : float, optional
        Fraction of data to use as external test set (default 0.2).
    scatterplot : bool, optional
        Whether to generate scatterplots and residual plots (default True).
    random_state : int, optional
        Random state for reproducibility in data splitting (default 10).
    scaling : bool, optional
        Whether to scale data (default False, usually PLS handles scaling internally).
    intervalpls : bool, optional
        If True, uses column 'Bands_selected' from df_config to train PLS only on selected bands.
        Indices refer to positions (0 = first band column), not wavelength values.

    Returns
    -------
    None

    Notes
    -----
    - Saves trained PLS models (or iPLS if intervalpls=True) using joblib.
    - Generates scatterplots of observed vs predicted values for train and external sets.
    - Generates residual plots (histograms and Top 10 per sample).
    - Generates X loadings plots, VIP plots, and band influence plots.
    """

       # ---------- 1. Create root save folder ----------
    save_path_root = os.path.join(save_path, "Results_definitive_pls_models")
    os.makedirs(save_path_root, exist_ok=True)

     # ---------- 2. Iterate over config rows ----------
    for _, row in df_config.iterrows():

        trait          = row['Trait']
        metric         = row['Metric']
        n_components   = int(row['Number of Components'])
        bands_selected = row.get('Bands_selected', None)

        # Create folder for the trait
        trait_dir = os.path.join(save_path_root, trait)
        os.makedirs(trait_dir, exist_ok=True)

        # ---------- 2a. Filter long DataFrame ----------
        df_filtered = df_long[df_long['Metric'] == metric]

        # ---------- 2a. Filter long DataFrame ----------
        df_pivot = (
            df_filtered
            .pivot_table(index=[index_column, trait],
                         columns='Band',
                         values='Value',
                         aggfunc='first')
            .reset_index()
        )


        start_col = 2   # columns after [ID, trait]
        full_band_cols = df_pivot.columns[start_col:]

        X = df_pivot.iloc[:, start_col:].values
        bands= full_band_cols                      
        y = df_pivot[trait].values

        # ---------- 2d. scaling ---------- PLS already applies scaling
        # if scaling:
        #     scaler = StandardScaler()
        #     X_scaled = scaler.fit_transform(X)
        #     scaler_filename = f"Scaler_{trait}_{metric}.pkl"
        #     joblib.dump(scaler, os.path.join(trait_dir, scaler_filename))
        #     print(f"✅ Scaler saved: {scaler_filename}")
        # else:
        #     X_scaled = X

        
        # ---------- 2b. Select bands for interval PLS ----------
        if intervalpls and not isinstance(bands_selected, float):
            # bands_selected can come as a list or a string → use ast.literal_eval
            sel_idx = row['Bands_selected']
            if isinstance(sel_idx, str):
                sel_idx = ast.literal_eval(sel_idx)

            # convert relative indices to absolute positions in the DataFrame
            abs_cols = [start_col + int(i) for i in sel_idx]
            X_ipls = df_pivot.iloc[:, abs_cols].values 
            bands= full_band_cols[sel_idx]      # for plots

            pls = PLSRegression(n_components=n_components)
            pls.fit(X_ipls, y)
            model_filename = f"iPLS_{trait}_{metric}_{n_components}comp.pkl"
        else:
            pls = PLSRegression(n_components=n_components)
            pls.fit(X, y)
            model_filename = f"PLS_{trait}_{metric}_{n_components}comp.pkl"
            
        joblib.dump(pls, os.path.join(trait_dir, model_filename))
        print(f"✅ Model saved: {model_filename}")

        # ---------- 2c. Scatterplots and residuals ----------

        if scatterplot:

            X_train, X_external, y_train, y_external, idx_train, idx_external = at.train_test_split(
                X, y, test_size=test_size, train_size=1-test_size, sampler="kennard_stone", random_state=random_state, return_indices=True)


            if intervalpls and not isinstance(bands_selected, float):
                X_ipls = df_pivot.iloc[:, abs_cols].values
                X_train = X_ipls[idx_train]
                X_external = X_ipls[idx_external]
                y_train = y[idx_train]
                y_external = y[idx_external]


            
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train, y_train)
            y_train_pred = pls.predict(X_train)
            y_external_pred = pls.predict(X_external)

            # External metrics
            r2_external = r2_score(y_external, y_external_pred)
            rmse_external = np.sqrt(mean_squared_error(y_external, y_external_pred))

             # You also need to provide this R² y RMSE de Calibration y Cross-Validation
            r2_calibration = row['Train R² Mean']  # You also need to provide this
            r2_cv = row['Test R² Mean']   # You also need to provide this
            rmse_calibration = row['Train RMSE Mean']  # You also need to provide this
            rmse_cv = row['Test RMSE Mean'] # You also need to provide this

            # Scatterplot
            plt.figure(figsize=(7, 7))
            plt.scatter(y_train, y_train_pred, label="Train", color="blue", alpha=0.7)
            plt.scatter(y_external, y_external_pred, label="External", color="red", alpha=0.7)

            # Reference lines
            min_val = min(y.min(), y_train_pred.min(), y_external_pred.min())
            max_val = max(y.max(), y_train_pred.max(), y_external_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")

            # Plot configuration
            plt.xlabel("Observed Values")
            plt.ylabel("Predicted Values")
            # plt.title(f"Scatterplot: {trait} ({metric}) - {n_components} Components")
            trait_label = "Fiber" if trait == "Fibber" else trait
            plt.title(f"{trait_label}")
            plt.legend()
            plt.grid(True)

            # Add metric text results
            metrics_text = (
                f"R² Calibration: {r2_calibration:.2f}\n"
                f"R² Cross Validation: {r2_cv:.2f}\n"
                f"R² External: {r2_external:.2f}\n"
                f"RMSE Calibration: {rmse_calibration:.2f}\n"
                f"RMSE Cross Validation: {rmse_cv:.2f}\n"
                f"RMSE External: {rmse_external:.2f}"
            )

            plt.gca().text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes,
                        horizontalalignment='right', verticalalignment='bottom',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

            # Save
            plot_filename = f"Scatter_{trait}_{metric}_{n_components}comp.png"
            plt.savefig(os.path.join(trait_dir, plot_filename), bbox_inches='tight', dpi=300)
            plt.close()
            print(f"📊 Scatter plot saved: {plot_filename}")

            # Residual histograms
            residuals_train = y_train - y_train_pred
            residuals_external = y_external - y_external_pred

            plt.figure(figsize=(8, 6))
            plt.hist(residuals_train, bins=20, color='lightblue', edgecolor='black', label='Train Residuals')
            plt.hist(residuals_external, bins=20, color='lightcoral', edgecolor='black', label='External Residuals')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title(f"Residuals Distribution: {trait} ({metric})")
            plt.legend()
            plt.grid(True)

            # Save
            residuals_histogram_filename = f"Residuals_Histogram_{trait}_{metric}_{n_components}comp.png"
            plt.savefig(os.path.join(trait_dir, residuals_histogram_filename))
            plt.close()

            # Individual residuals top 10
            residuals_combined = np.concatenate([residuals_train, residuals_external])
            individual_ids = np.concatenate([df_pivot[index_column].values, df_pivot[index_column].values])

            # Order
            sorted_indices = np.argsort(np.abs(residuals_combined))[::-1][:10]
            top_residuals = residuals_combined[sorted_indices]
            top_individual_ids = individual_ids[sorted_indices]

            # Bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(np.arange(len(top_residuals)), top_residuals, color='lightcoral')
            plt.axhline(0, color='black', linestyle='--', linewidth=2)

            # Show top 10
            plt.xticks(np.arange(len(top_residuals)), top_individual_ids, rotation=90)
            plt.xlabel('Individual (ID)')
            plt.ylabel('Residual')
            plt.title(f"Top 10 Residuals per Individual: {trait} ({metric})")
            plt.grid(True)

            # Save
            residuals_bars_filename = f"Top_10_Residuals_Bars_{trait}_{metric}_{n_components}comp.png"
            plt.savefig(os.path.join(trait_dir, residuals_bars_filename))
            plt.close()


            print(f"📉 Residuals plots saved: {residuals_histogram_filename} & {residuals_bars_filename}")

            # Loading plot
            for comp in range(n_components):
                loadings = pls.x_loadings_[:, comp]
                plt.plot(bands, loadings, label=f'LV {comp+1}')

            plt.xticks(rotation=45)
            plt.xlabel('Spectral Bands')
            plt.ylabel('Loadings')
            plt.title(f'X Loadings per Latent Variable: {trait} ({metric})')
            plt.legend(title='Latent Variables')
            plt.grid(True)
            plt.tight_layout()

            loadings_plot_filename = f"XLoadings_{trait}_{metric}_{n_components}comp.png"
            plt.savefig(os.path.join(trait_dir, loadings_plot_filename))
            plt.close()

            print(f"📈 X Loadings plot saved: {loadings_plot_filename}")

           # Calculate vips 
            vips = calculate_vip(pls)

            # Plot VIP with green and red colors
            colors = ['green' if v >= 1 else 'red' for v in vips]

            plt.figure(figsize=(12, 6))
            plt.bar(bands, vips, color=colors)
            plt.axhline(1.0, color='black', linestyle='--', linewidth=1)
            plt.xlabel('Spectral Bands')
            plt.ylabel('VIP Score')
            plt.title(f'VIP Scores - {trait} ({metric})')
            plt.tight_layout()

            # Save VIP plots
            vip_plot_filename = f"VIP_{trait}_{metric}_{n_components}comp.png"
            plt.savefig(os.path.join(trait_dir, vip_plot_filename))
            plt.close()

            print(f"🌟 VIP plot saved: {vip_plot_filename}")

            # Formatting to matplotlib
            bands = [float(b) for b in bands]  # Conver to float for plotting
            influence = np.asarray(pls.coef_).ravel()

            plt.figure(figsize=(12, 6))
            plt.plot(bands, influence, color='black', linewidth=2, label='Coefficient')

            #Only fill for PLS
            if intervalpls is False:

                plt.fill_between(bands, 0, influence, where=(influence > 0), interpolate=True, color='green', alpha=0.3, label='Positive Influence')
                plt.fill_between(bands, 0, influence, where=(influence < 0), interpolate=True, color='red', alpha=0.3, label='Negative Influence')

            num_ticks = 4
            tick_positions = np.linspace(min(bands), max(bands), num_ticks)
            plt.xticks(tick_positions, [f"{int(tick)}" for tick in tick_positions])

            plt.xlabel('Spectral Bands')
            plt.ylabel('Coefficient')
            plt.title(f'Band Influence on Trait: {trait} ({metric})')
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.legend()
            plt.tight_layout()

            coef_plot_filename = f"BandInfluence_{trait}_{metric}_{n_components}comp.png"
            plt.savefig(os.path.join(trait_dir, coef_plot_filename))
            plt.close()

            print(f"📉 Band influence plot saved: {coef_plot_filename}")

    print("✅ All models, scalers, and plots saved successfully.")

def calculate_vip(model):
    """
    Calculate the Variable Importance in Projection (VIP) scores for a fitted PLS model.

    Parameters
    ----------
    model : PLSRegression
        A fitted PLSRegression model from scikit-learn.

    Returns
    -------
    vips : ndarray
        Array of VIP scores for each predictor variable.
    
    Notes
    -----
    VIP scores summarize the contribution of each predictor to the PLS model.
    A VIP score > 1 generally indicates an important variable.
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
    return vips

def pls_validation(df_long, traits, preprocessed_metrics, save_path, index_column='ID', n_components_range=range(1, 11),
                    n_splits=7, n_repeats=50, external_split=True, test_size=0.2, random_state=10, scaling=False):
    
    # =============================================================================
    # General Description:
    # --------------------
    # This function trains and validates Partial Least Squares Regression (PLSR) models 
    # using spectral data in long format. It allows you to:
    #   - Evaluate different preprocessing methods (RAW, SNV, MSC, SG, etc.).
    #   - Test different numbers of PLS components.
    #   - Perform repeated cross-validation (Repeated KFold).
    #   - Optionally perform an external train/test split using Kennard-Stone. 
    #   - Compute performance metrics such as RMSE and R² for train, test, and external sets.
    #   - Save results into summary tables and generate boxplots and line plots.
    #
    # Essentially, it is used to evaluate the performance of PLS models depending on:
    #   - the number of components,
    #   - the preprocessing method applied,
    #   - and the target trait to predict.
    #
    # =============================================================================
    # Arguments:
    #
    # df_long : pd.DataFrame
    #     Long-format DataFrame containing columns: sample ID, Band, Value, Metric, and traits.
    #
    # traits : list of str
    #     List of trait column names (dependent variables) to be predicted.
    #
    # preprocessed_metrics : list of str
    #     List of preprocessing metrics to include (e.g., RAW, SNV, MSC, ...).
    #
    # save_path : str
    #     Directory path where the results (summaries and plots) will be saved.
    #
    # index_column : str, default 'ID'
    #     Column name that uniquely identifies each sample.
    #
    # n_components_range : range, default range(1, 11)
    #     Range of PLS components to evaluate.
    #
    # n_splits : int, default 7
    #     Number of folds for cross-validation (KFold).
    #
    # n_repeats : int, default 50
    #     Number of repetitions for repeated cross-validation (Repeated KFold).
    #
    # external_split : bool, default True
    #     If True, an external train/test split is performed in addition to cross-validation.
    #
    # test_size : float, default 0.2
    #     Proportion of data reserved as external test set (e.g., 0.2 = 20%).
    #
    # random_state : int, default 10
    #     Random seed to ensure reproducibility of splits.
    #
    # scaling : bool, default False
    #     If True, standardizes spectral variables (mean=0, variance=1).
    #
    # =============================================================================

    start_column = len(traits) + 1  # Columns after traits
    
    # DataFrame to save results
    summary_df = pd.DataFrame()
    summary_box_df = pd.DataFrame()

    # Loop over pre-processing methods employed

    for metric in preprocessed_metrics:
        df_long_filtered = df_long[df_long['Metric'] == metric]
        df_pivot = df_long_filtered.pivot_table(
            index=[index_column] + traits, 
            columns='Band', values='Value', aggfunc='first').reset_index()
        
        X = df_pivot.iloc[:, start_column:]
        if scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.to_numpy() 
        
        y_df = df_pivot[traits]  # Keep as df to avoid indexing problems
        
        if external_split:
            X_train, X_external, y_train, y_external = at.train_test_split(X_scaled, y_df, test_size=test_size,
                                                                            train_size=1-test_size, sampler="kennard_stone", random_state=random_state)
        else:
            X_train, y_train = X_scaled, y_df
            X_external, y_external = None, None

        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        # Loop over traits
        
        for trait in traits:
            # Inicializamos las listas para almacenar los resultados
            train_rmse_results, test_rmse_results, external_rmse_results = [], [], []
            train_r2_results, test_r2_results, external_r2_results = [], [], []
            
            trait_index = traits.index(trait)  # Obtain index column of the trait

            if external_split:
                y_train_trait = y_train[:, trait_index]  # Select only the y of the trait
                y_external_trait = y_external[:, trait_index]
            else:
                y_train_trait = df_pivot[trait]

            # Save results for each component
            summary_df_component=pd.DataFrame()
            summary_box_df_component=pd.DataFrame()

            # Loop over components range

            for n_components in n_components_range:
                pls = PLSRegression(n_components=n_components)
                train_rmse_fold, test_rmse_fold = [], []
                train_r2_fold, test_r2_fold = [], []

                for train_idx, test_idx in cv.split(X_train):
                    X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
                    y_train_cv, y_test_cv = y_train_trait[train_idx], y_train_trait[test_idx]

                    pls.fit(X_train_cv, y_train_cv)

                    y_train_pred = pls.predict(X_train_cv)
                    train_rmse_fold.append(np.sqrt(mean_squared_error(y_train_cv, y_train_pred)))
                    train_r2_fold.append(r2_score(y_train_cv, y_train_pred))

                    y_test_pred = pls.predict(X_test_cv)
                    test_rmse_fold.append(np.sqrt(mean_squared_error(y_test_cv, y_test_pred)))
                    test_r2_fold.append(r2_score(y_test_cv, y_test_pred))

                # Save results for each n component 
                train_rmse_results.append(train_rmse_fold)
                test_rmse_results.append(test_rmse_fold)
                train_r2_results.append(train_r2_fold)
                test_r2_results.append(test_r2_fold)
                
                # Train with all training data and predict on external set **
                if external_split:
                    pls.fit(X_train, y_train_trait)  
                    y_external_pred = pls.predict(X_external)  
                    external_rmse_results.append(np.sqrt(mean_squared_error(y_external_trait, y_external_pred)))
                    external_r2_results.append(r2_score(y_external_trait, y_external_pred))

                summary_data = [
                    {
                        'Trait': trait,
                        'Metric': metric,
                        'Number of Components': n_components,
                        'Dataset': dataset,
                        'RMSE': rmse,
                        'R²': r2,
                        'Random_state': random_state
                    }
                    for i in range(len(train_rmse_fold))
                    for dataset, rmse, r2 in zip(
                        ['Train', 'Test'],
                        [train_rmse_fold[i], test_rmse_fold[i]],
                        [train_r2_fold[i], test_r2_fold[i]]
                    )
                ]
                summary_data=pd.DataFrame(summary_data)
                summary_box_df_component = pd.concat([summary_box_df_component, summary_data], ignore_index=True)
                summary_box_df = pd.concat([summary_box_df, summary_data], ignore_index=True)
                
            # Convert to np to improve the data managment
            train_rmse_results, test_rmse_results = np.array(train_rmse_results), np.array(test_rmse_results)
            train_r2_results, test_r2_results = np.array(train_r2_results), np.array(test_r2_results)
            
            # Convert to np to improve the data managment
            if external_split:
                external_rmse_results = np.array(external_rmse_results)
                external_r2_results = np.array(external_r2_results)
            
            # Summary 
            for n_components in n_components_range:
                summary_row = {
                                    'Trait': trait,
                                    'Metric': metric,
                                    'Number of Components': n_components,
                                    'Train RMSE Mean': np.mean(train_rmse_results[n_components-1]),
                                    'Test RMSE Mean': np.mean(test_rmse_results[n_components-1]),
                                    'External RMSE Mean': external_rmse_results[n_components-1] if external_split else None,
                                    'Train R² Mean': np.mean(train_r2_results[n_components-1]),
                                    'Test R² Mean': np.mean(test_r2_results[n_components-1]),
                                    'External R² Mean': external_r2_results[n_components-1] if external_split else None
                                }
                summary_row=pd.DataFrame([summary_row])
                summary_df_component=pd.concat([summary_df_component, summary_row], ignore_index=True)
                summary_df = pd.concat([summary_df, summary_row], ignore_index=True)

            
            # Create trait folder
            variable_folder = os.path.join(save_path, trait)
            os.makedirs(variable_folder, exist_ok=True)
            
            # Convert to dataframe the summaries dictionaries
            summary_df = pd.DataFrame(summary_df)
            summary_box_df = pd.DataFrame(summary_box_df)

            # Exportar .txt
            summary_df.to_csv(os.path.join(save_path, 'summary_df.txt'), sep='\t', index=False)
            summary_box_df.to_csv(os.path.join(save_path, 'summary_box_df.txt'), sep='\t', index=False)
            
            # Boxplot figure
            plt.figure(figsize=(12, 6))

            # A: Boxplot  RMSE
            plt.subplot(1, 2, 1)
            sns.boxplot(x="Number of Components", y="RMSE", hue="Dataset", data=summary_box_df_component, palette="Set1", width=0.8)
            plt.title('Boxplot of RMSE')
            plt.xlabel('Number of Components')
            plt.ylabel('RMSE')

            # B: Boxplot R²
            plt.subplot(1, 2, 2)
            sns.boxplot(x="Number of Components", y="R²", hue="Dataset", data=summary_box_df_component, palette="Set1", width=0.8)
            plt.title('Boxplot of R²')
            plt.xlabel('Number of Components')
            plt.ylabel('R²')

            plt.tight_layout()
            plt.savefig(os.path.join(variable_folder, f'{trait}_{metric}_boxplot_rmse_r2_sns.png'))
    

            # Line plot figure 
            plt.figure(figsize=(14, 6))

            # RMSE
            plt.subplot(1, 2, 1)
            plt.plot(summary_df_component['Number of Components'], summary_df_component['Train RMSE Mean'], label='Train RMSE', marker='o')
            plt.plot(summary_df_component['Number of Components'], summary_df_component['Test RMSE Mean'], label='Test RMSE', marker='o')
            if external_split:
                plt.plot(summary_df_component['Number of Components'], summary_df_component['External RMSE Mean'], label='External RMSE', marker='o')
            plt.title('Line Plot of RMSE')
            plt.xlabel('Number of Components')
            plt.ylabel('RMSE')
            plt.legend()

            # R² 
            plt.subplot(1, 2, 2)
            plt.plot(summary_df_component['Number of Components'], summary_df_component['Train R² Mean'], label='Train R²', marker='o')
            plt.plot(summary_df_component['Number of Components'], summary_df_component['Test R² Mean'], label='Test R²', marker='o')
            if external_split:
                plt.plot(summary_df_component['Number of Components'], summary_df_component['External R² Mean'], label='External R²', marker='o')
            plt.title('Line Plot of R²')
            plt.xlabel('Number of Components')
            plt.ylabel('R²')
            plt.legend()

            # Save
            plt.tight_layout()
            plt.savefig(os.path.join(variable_folder, f'{trait}_{metric}_line_plot_rmse_r2.png'))
            plt.close()

def pls_interval_validation(df_long, trait, preprocessed_metric, save_path, index_column='ID', n_components=11,
                    n_splits=8, n_repeats=10, external_split=True, intervals_sizes=[5,10], max_number_interval_selected_tested=10, test_size=0.2,
                      random_state=10, n_jobs=-1, constraint=1, full_mode=True):
    

    # =============================================================================
    # Interval Partial Least Squares (iPLS) with Complexity Constraint
    # =============================================================================
    # Description:
    # ------------
    # This function performs Interval Partial Least Squares (iPLS) regression 
    # with a forward selection strategy and a complexity constraint.
    # - It evaluates all possible interval combinations up to the maximum defined 
    #   by `max_number_interval_selected_tested`.
    # - The process continues until all possible intervals are tested if full_mode=True, 
    #   providing a full/exhaustive evaluation. If full_mode=False, the selection stops 
    #   when adding intervals does not improve prediction error.
    # - It outputs all evaluated results, not just the best model.
    # - The "complexity constraint" ensures that the number of components only 
    #   increases if the prediction error is sufficiently reduced 
    #   (controlled by `constraint` parameter).
    # - Recommended for HPC environments if many combinations are to be tested.
    #
    # Arguments:
    # ----------
    # df_long : pd.DataFrame
    #     Long-format DataFrame with columns: sample ID, Band, Value, Metric, and trait.
    #
    # trait : str
    #     Name of the trait column (dependent variable) to predict.
    #
    # preprocessed_metric : str
    #     Preprocessing method to use (e.g., RAW, SNV, MSC, ...).
    #
    # save_path : str
    #     Directory path where the results will be saved.
    #
    # index_column : str, default 'ID'
    #     Column name that uniquely identifies each sample.
    #
    # n_components : int, default 11
    #     Maximum number of PLS components to test for each interval combination.
    #
    # n_splits : int, default 8
    #     Number of folds for cross-validation.
    #
    # n_repeats : int, default 10
    #     Number of repetitions for repeated cross-validation.
    #
    # external_split : bool, default True
    #     If True, performs an external train/test split using Kennard-Stone.
    #
    # intervals_sizes : list of int, default [5, 10]
    #     Sizes of intervals (number of spectral variables per interval) to test.
    #
    # max_number_interval_selected_tested : int, default 10
    #     Maximum number of intervals to select during the forward selection.
    #
    # test_size : float, default 0.2
    #     Proportion of data reserved as external set.
    #
    # random_state : int, default 10
    #     Seed for reproducibility.
    #
    # n_jobs : int, default -1
    #     Number of CPU cores to use for parallelization (if implemented).
    #
    # constraint : float, default 1
    #     Complexity constraint controlling the allowed increase of components
    #     only if prediction error is sufficiently reduced.
    #
    # full_mode : bool, default True
    #     If True, perform full evaluation of all interval combinations, ignoring
    #     temporary increases in RMSE. If False, stop adding intervals when the
    #     prediction error no longer improves.


    # For reading about ipls #https://wiki.eigenvector.com/index.php?title=Interval_PLS_(IPLS)_for_Variable_Selection #
    # =============================================================================
        
    # Create the necessary directories if they don't already exist
    ipls_results_directory=os.path.join(save_path,"iPLS_results")
    if not os.path.exists(ipls_results_directory):
        os.makedirs(ipls_results_directory)

    mem_log = []

    # Filter for preprocessing method and trait
    df_long_filtered = df_long[df_long['Metric'] == preprocessed_metric]
    df_pivot = df_long_filtered.pivot_table(
        index=[index_column] + [trait], 
        columns='Band', values='Value', aggfunc='first'
    ).reset_index()
        
    X = df_pivot.drop(columns=[index_column, trait])
    y_df = df_pivot[trait]
    
    #Divide in external split

    if external_split:
        X_train, X_external, y_train_trait, y_external_trait = at.train_test_split(X, y_df, test_size=test_size,
                                                                        train_size=1-test_size, sampler="kennard_stone", random_state=random_state)
    else:
        X_train, y_train_trait = X, y_df
        X_external, y_external_trait = None, None


    # Convert to numpy to work in numpy
    X_train_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
    y_train_np = y_train_trait.values if hasattr(y_train_trait, 'values') else np.array(y_train_trait)

    if X_external is not None:
        X_external_np = X_external.values if hasattr(X_external, 'values') else np.array(X_external)
    else:
        X_external_np = None

    if y_external_trait is not None:
        y_external_np = y_external_trait.values if hasattr(y_external_trait, 'values') else np.array(y_external_trait)
    else:
        y_external_np = None

    # Set CV
    cv = RepeatedKFold(
    n_splits=n_splits,          
    n_repeats=n_repeats,         
    random_state=random_state
)
    
    # List to save results
    results_rows = []

    # Number of bands
    n_features = X_train_np.shape[1]
    
    # Iterate over different interval sizes to test
    for interval_size in intervals_sizes:            # e.g. [10, 20, 30]
        
        n_intervals = n_features // interval_size
        max_intervals_to_use = min(max_number_interval_selected_tested,
                                n_intervals)

        # Columns per interval
        interval_cols = [
            list(range(i*interval_size,
                    min((i+1)*interval_size, n_features)))
            for i in range(n_intervals)
        ]

        selected_intervals  = []      
        remaining_intervals = list(range(n_intervals))

        # Variables for Wold
        best_val_rmse = np.inf
        best_val_r2   = None
        best_train_rmse = None
        best_train_r2   = None
        best_idx      = None
        best_pls      = None
        best_cols     = None
        best_n_components = float('inf')

        # ---------- Forward-selection iPLS ----------
        for k in range(1, max_intervals_to_use + 1):


            print(f"Trait: {trait}, Interval: {interval_size}, K: {k}, constraint: {constraint}",flush=True)

            best_idx = None
                        
            # If full_mode is True, perform a full evaluation of all interval combinations, 
            # ignoring temporary increases in RMSE. 
            # If full_mode is False, stop adding intervals when the prediction error no longer improves.
            if full_mode:
                best_val_rmse = np.inf

            # Iterates over component range if it is possible due to the number of bands 
            for n_component in range(1, min(n_components+1, (k*interval_size)+1)):

                for cand in remaining_intervals:

                    cand_set = selected_intervals + [cand]
                    cols = [c for idx in cand_set for c in interval_cols[idx]]

                    X_sel = X_train_np[:, cols]

                    val_rmse_folds = []
                    train_rmse_folds = []
                    val_r2_folds = []
                    train_r2_folds = []

                    for train_index, val_index in cv.split(X_sel):
                        X_tr, X_val = X_sel[train_index, :], X_sel[val_index, :]
                        y_tr, y_val = y_train_np[train_index], y_train_np[val_index]

                        pls_tmp = PLSRegression(n_components=n_component, scale=True)
                        pls_tmp.fit(X_tr, y_tr)

                        y_tr_pred = pls_tmp.predict(X_tr)
                        y_val_pred = pls_tmp.predict(X_val)

                        train_rmse_folds.append(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
                        val_rmse_folds.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

                        train_r2_folds.append(r2_score(y_tr, y_tr_pred))
                        val_r2_folds.append(r2_score(y_val, y_val_pred))

                    val_rmse = np.mean(val_rmse_folds)
                    val_r2 = np.mean(val_r2_folds)
                    train_rmse = np.mean(train_rmse_folds)
                    train_r2 = np.mean(train_r2_folds)

                    #Here the constraint is applied

                    if val_rmse <= best_val_rmse * constraint or (val_rmse <= best_val_rmse and n_component <= best_n_components):
                        best_val_rmse = val_rmse
                        best_val_r2 = val_r2
                        best_train_rmse = train_rmse
                        best_train_r2 = train_r2
                        best_idx = cand
                        best_pls = pls_tmp
                        best_cols = cols
                        best_n_components = n_component

            # If there is improvement
            if best_idx is not None:
                selected_intervals.append(best_idx)
                remaining_intervals.remove(best_idx)

                X_train_sel = X_train_np[:, best_cols]
                X_external_sel = X_external_np[:, best_cols]

                best_pls.fit(X_train_sel, y_train_np)
                y_ext_pred = best_pls.predict(X_external_sel)

                ext_rmse = np.sqrt(mean_squared_error(y_external_np, y_ext_pred))
                ext_r2 = r2_score(y_external_np, y_ext_pred)

                results_rows.append({
                    'n_components': best_n_components,
                    'interval_size': interval_size,
                    'number_selected_interval': k,
                    'selected_intervals': selected_intervals.copy(),
                    'selected_bands': best_cols.copy(),
                    'train_r2': best_train_r2,
                    'val_r2': best_val_r2,
                    'external_r2': ext_r2,
                    'train_rmse': best_train_rmse,
                    'val_rmse': best_val_rmse,
                    'external_rmse': ext_rmse,
                    'constraint': constraint,
                    'overfit_ratio': best_val_rmse / best_train_rmse
                })

                print(results_rows[-1])

            else:
                # If there is not improvement
                print("Local minimum RMSE achieved")
                break

    # List to df
    results_df = pd.DataFrame(results_rows)

    # Save
    results_df.to_csv(f"{ipls_results_directory}/{trait}_{constraint}_results_ipls.txt", sep='\t', index=False)

def apply_saved_pls_models(df_long, df_config, index_column='ID', save_dir=None, ipls=False):

    # =============================================================================

    # Function: ef.train_and_save_pls_models
    #
    # Description:
    # This function trains and saves PLS (Partial Least Squares) models for each row in a configuration DataFrame (`df_config`).
    # It can handle standard PLS or interval-PLS (iPLS), using only selected spectral bands if desired.
    # For each model, it optionally generates scatter plots, residual plots, X loadings plots, VIP plots, and band influence plots.
    #
    # Main steps:
    # 1. Create a root folder to save all results (`Results_definitive_pls_models`).
    # 2. Loop over each row of `df_config`:
    #    - Extract trait, preprocessing metric, number of components, and selected bands (for iPLS).
    #    - Filter `df_long` by the metric and pivot to wide format (samples x bands).
    #    - If `intervalpls=True`, use only the bands specified in `Bands_selected`.
    #    - Train the PLS model with the specified number of components.
    #    - Save the trained model to disk.
    # 3. If `scatterplot=True`:
    #    - Perform an external train/test split (Kennard-Stone).
    #    - Generate scatter plots of predicted vs true values for training and external sets.
    #    - Compute and display RMSE and R² for calibration, cross-validation, and external sets.
    #    - Plot residuals (histogram and top 10 individuals).
    #    - Plot X loadings per latent variable.
    #    - Plot VIP (Variable Importance in Projection) scores.
    #    - Plot band influence (PLS coefficients) with positive/negative areas highlighted.
    #
    # Arguments:
    # df_long : pd.DataFrame
    #     Long-format spectral data with columns [ID, Trait, Metric, Band, Value].
    # df_config : pd.DataFrame
    #     Each row contains model parameters: Trait, Metric, Number of Components, Bands_selected, and performance metrics.
    # save_path : str
    #     Root folder to save models, plots, and results.
    # index_column : str, default 'ID'
    #     Column that uniquely identifies each sample.
    # test_size : float, default 0.2
    #     Proportion of data reserved for the external set.
    # scatterplot : bool, default True
    #     If True, generate plots (scatter, residuals, loadings, VIP, band influence).
    # random_state : int, default 10
    #     Random seed for reproducibility.
    # scaling : bool, default False
    #     If True, scale spectral variables (mean=0, variance=1). PLS internally standardizes by default.
    # intervalpls : bool, default False
    #     If True, train models only using the selected bands in `Bands_selected` column (iPLS).
    #
    # Notes:
    # - The function creates subfolders for each trait.
    # - Band indices in `Bands_selected` are relative positions (0 = first band column after ID/trait).
    # - All models, plots, and optionally scalers are saved to disk.

    # =============================================================================


    results = []

    # 0 col is index column bands start on 1
    start_col = 1

    # Iteratos on rows of df_config file
    for _, row in df_config.iterrows():
        trait = row['Trait']
        metric = row['Metric']
        model_path = row['model_path']
        scaler_path = row['scale_path']
        bands_selected = row.get('Bands_selected', None)

        

        # Obtain band selected
        if isinstance(bands_selected, str):
            try:
                bands_selected = ast.literal_eval(bands_selected)
            except Exception as e:
                print(f"⚠️ No se pudo parsear Bands_selected para {trait} ({metric}): {e}")
                bands_selected = None

        # Filter preprocessing method
        df_filtered = df_long[df_long['Metric'] == metric]

        # Arrange df
        df_pivot = df_filtered.pivot_table(
            index=[index_column],
            columns='Band', values='Value', aggfunc='first'
        ).reset_index()

        ids = df_pivot[index_column].values

        # Obtain bands name in nm
        full_band_cols = list(df_pivot.columns[start_col:])
        if ipls and not isinstance(bands_selected, float):
            print(trait, "iPLS")
            # bands_selected is a list of indices relative to full_band_cols

            # Chck indexes
            max_idx = len(full_band_cols) - 1
            sel_idx_valid = [int(i) for i in bands_selected if 0 <= int(i) <= max_idx]

            if len(sel_idx_valid) != len(bands_selected):
                print(f"⚠️ Some indices in Bands_selected for {trait} ({metric}) are out of range. Breaking the loop.")
                break


            # From relative to absolute
            abs_cols = [start_col + i for i in sel_idx_valid]

        

            # Extract values of this columns
            X = df_pivot.iloc[:, abs_cols].values
            # X maintains the order as provided in the input. Do not change it.
        else:
            # If not using iPLS or no bands_selected provided, use all available bands
            print(trait, "PLS")
            X = df_pivot.drop(columns=[index_column]).values
            bands = full_band_cols

        # Scaling
        if isinstance(scaler_path, str) and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
            print(f"📏 Scaled {trait} ({metric})")
        else:
            print(f"⚠️ No scale for {trait} ({metric})")

        # Load model
        if not os.path.exists(model_path):
            print(f"❌ Model not found for {trait} ({metric})")
            continue
        model = joblib.load(model_path)
        print(f"📦 Model loaded for {trait} ({metric})")

        # Predict
        y_pred = model.predict(X).ravel()

        # Compute DModX
        dmodx = compute_dmodx(model, X)

        df_result = pd.DataFrame({
            index_column: ids,
            f'Pred_{trait}': y_pred,
            f'{trait}_dmodx': dmodx
        })

        results.append(df_result)

    # Merge results
    if results:
        df_final = results[0]
        for df in results[1:]:
            df_final = df_final.merge(df, on=index_column)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "predictions_pls.txt")
            df_final.to_csv(save_path, sep='\t', index=False)
            print(f"📝 Predictions save in: {save_path}")

        return df_final
    else:
        print("⚠️ No predictions generated")
        return pd.DataFrame()

def cow_warp(reference, sample, n_segments=5, slack=2):
    """
    Simplified implementation of Correlation Optimized Warping (COW) for 1D spectral data.

    Parameters:
    -----------
    reference : 1D array
        Reference spectrum to align to.
    sample : 1D array
        Spectrum to be warped/aligned.
    n_segments : int, default=5
        Number of segments to divide the spectra into.
    slack : int, default=2
        Maximum allowed stretching/compression per segment (in indices).

    Returns:
    --------
    aligned : 1D array
        Warped version of the sample spectrum aligned to the reference.

    Description:
    ------------
    The function splits the reference spectrum into `n_segments` segments. 
    For each segment, it searches within the `slack` range for the best alignment 
    of the corresponding sample segment, using Pearson correlation as the criterion. 
    The aligned segments are then combined to produce the final warped spectrum.
    """
    # Length of reference spectrum
    len_ref = len(reference)
    # Segment length for each piece
    seg_len = len_ref // n_segments
    # Initialize output aligned spectrum
    aligned = np.zeros_like(sample)

    # Iterate over each segment
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else len_ref  # last segment may differ

        best_corr = -np.inf
        best_segment = None

        # Search for the best alignment within the allowed slack
        for shift in range(-slack, slack + 1):
            s_start = max(start + shift, 0)
            s_end = min(end + shift, len(sample))

            ref_seg = reference[start:end]
            samp_seg = sample[s_start:s_end]

            # Ensure equal lengths before correlation
            if len(samp_seg) == len(ref_seg):
                corr = np.corrcoef(ref_seg, samp_seg)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_segment = samp_seg

        # Insert the best segment found into the aligned spectrum
        aligned[start:end] = best_segment

    return aligned

def apply_cow_to_multiple_metrics(df_all_results, preprocessing_metrics, sample_id_col='ID',
                                  n_segments=5, slack=2, reference='mean'):
    """
    Applies COW (Correlation Optimized Warping) to multiple spectral metrics in a DataFrame,
    overwriting the original columns with aligned values.

    Parameters:
    -----------
    df_all_results : pd.DataFrame
        DataFrame containing columns 'Band', sample_id_col, and spectral metrics.
    preprocessing_metrics : list
        List of columns (spectral metrics) to apply COW to.
    sample_id_col : str, default 'ID'
        Column name that identifies each sample.
    n_segments : int, default 5
        Number of segments to divide each spectrum into. More segments = finer alignment,
        but higher risk of overfitting.
    slack : int, default 2
        Maximum allowed stretch/compression per segment. Higher slack = more flexibility,
        but may distort the signal.
    reference : str, default 'mean'
        Reference spectrum to align against. Can be:
        - 'mean' : mean spectrum of all samples
        - 'median' : median spectrum of all samples
        - 'sample:<ID>' : use a specific sample as reference (replace <ID> with the sample ID)

    Returns:
    --------
    pd.DataFrame
        A copy of df_all_results with the specified spectral metrics aligned using COW.
        Original columns are overwritten with the aligned spectra.
    """
    
    # Copy the DataFrame to avoid modifying the original
    df_result = df_all_results.copy()
    
    for metric in preprocessing_metrics:
        # Pivot the DataFrame to have samples as columns and bands as rows
        df_pivot = df_result.pivot(index='Band', columns=sample_id_col, values=metric)
        bands = df_pivot.index.values

        # Select the reference spectrum
        if reference == 'mean':
            ref_spectrum = df_pivot.mean(axis=1).values
        elif reference == 'median':
            ref_spectrum = df_pivot.median(axis=1).values
        elif reference.startswith("sample:"):
            sample_id = reference.split(":")[1]
            if sample_id not in df_pivot.columns:
                raise ValueError(f"Sample ID '{sample_id}' not found in data.")
            ref_spectrum = df_pivot[sample_id].values
        else:
            raise ValueError("reference must be 'mean', 'median' or 'sample:<ID>'")

        # Align each sample individually using COW
        aligned_data = {}
        for sample_id in df_pivot.columns:
            sample_spectrum = df_pivot[sample_id].values
            aligned = cow_warp(ref_spectrum, sample_spectrum, n_segments=n_segments, slack=slack)
            aligned_data[sample_id] = aligned

        # Reconstruct aligned DataFrame and melt back to long format
        df_aligned = pd.DataFrame(aligned_data, index=bands)
        df_aligned.index.name = 'Band'
        df_aligned = df_aligned.reset_index().melt(
            id_vars='Band', var_name=sample_id_col, value_name=metric
        )

        # Replace the original column with the aligned values
        df_result = df_result.drop(columns=[metric])
        df_result = df_result.merge(df_aligned, on=['Band', sample_id_col], how='left')

    return df_result


def compute_dmodx(model, X):
    """
    Compute DModX (Distance to the Model in X-space).

    DModX measures the residual distance between the original X variables
    and their reconstruction from the PLS latent variable model.
    It reflects how well each sample is represented by the PLS model.

    Higher values indicate samples that are poorly described by the model.
    """

    # Scores
    T = model.transform(X)

    # X loadings
    P = model.x_loadings_

    # Reconstruction of X
    X_hat = np.dot(T, P.T)

    # Residual matrix
    residual = X - X_hat

    # DModX per sample
    dmodx = np.sqrt(np.sum(residual**2, axis=1))

    return dmodx