# %%
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import seaborn as sns
import os

# %%

# Path to data
path2real = "/home/nestorask/Projects/Machine_learning_tutorial_HBio2025_server/data/real/"

path2synthetic = "/home/nestorask/Projects/Machine_learning_tutorial_HBio2025_server/results/CTGAN_models/"

# Select train files
train_files = [path2real+f for f in os.listdir(
    path2real) if f.endswith('_training_data.csv')]


# %%
real_data_file = train_files[0]
for real_data_file in train_files:

    print(
        f"# Evaluating synthetic data from {os.path.basename(real_data_file)}")

    # Read real data
    real_data = pd.read_csv(real_data_file)

    # Read synthetic data
    synthetic_data_file = path2synthetic + \
        "Synthesizer_" + os.path.basename(real_data_file)
    synthetic_data = pd.read_csv(synthetic_data_file)

    # Drop columns from real data that are not in synthetic data
    real_data = real_data[synthetic_data.columns]

    # Make metadata
    metadata = Metadata.detect_from_dataframe(data=real_data)

    # Evaluating Real vs Synthetic data
    diagnostic = run_diagnostic(
        real_data=real_data,
        synthetic_data=synthetic_data, metadata=metadata
    )
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata
    )

    quality_reports_details = quality_report.get_details('Column Shapes')
    print(quality_reports_details)

    # Calculate correlation matrices
    corr_real = real_data.corr()
    corr_synthetic = synthetic_data.corr()

    # Perform hierarchical clustering on the correlation matrices
    linkage_real = linkage(corr_real, method='ward')
    linkage_synthetic = linkage(corr_synthetic, method='ward')

    # Plot dendrogram for real data correlation matrix
    plt.figure(figsize=(12, 10))
    dendrogram(linkage_real, labels=corr_real.columns, leaf_rotation=90)
    plt.title('Dendrogram of Real Data Correlation Matrix')
    plt.tight_layout()
    plt.savefig(synthetic_data_file.replace(
        ".csv", "_dendrogram_real_data.pdf"))
    plt.show()
    plt.close()

    # Plot dendrogram for synthetic data correlation matrix
    plt.figure(figsize=(12, 10))
    dendrogram(linkage_synthetic,
               labels=corr_synthetic.columns, leaf_rotation=90)
    plt.title('Dendrogram of Synthetic Data Correlation Matrix')
    plt.tight_layout()
    plt.savefig(synthetic_data_file.replace(
        ".csv", "_dendrogram_synthetic_data.pdf"))
    plt.show()
    plt.close()

    # Plot heatmap for real data correlation matrix
    # Mask to plot only the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr_real, dtype=bool))

    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_real, mask=mask, cmap='coolwarm',
                annot=True, fmt=".2f", square=True)
    plt.title('Heatmap of Real Data Correlation Matrix (Upper Triangle)')
    plt.tight_layout()
    plt.savefig(synthetic_data_file.replace(
        ".csv", "_heatmap_cor_real_data.pdf"))
    plt.show()
    plt.close()

    # Plot heatmap for synthetic data correlation matrix
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_synthetic, mask=mask, cmap='coolwarm',
                annot=True, fmt=".2f", square=True)
    plt.title('Heatmap of Synthetic Data Correlation Matrix (Upper Triangle)')
    plt.tight_layout()
    plt.savefig(synthetic_data_file.replace(
        ".csv", "_heatmap_cor_synthetic_data.pdf"))
    plt.show()
    plt.close()

    # Calculate correlation between flattened correlation matrices
    # Remove diagonal elements from the correlation matrices
    # Get the upper triangle of the correlation matrices
    upper_tri_real = corr_real.where(
        np.triu(np.ones(corr_real.shape), k=1).astype(bool))
    upper_tri_synthetic = corr_synthetic.where(
        np.triu(np.ones(corr_synthetic.shape), k=1).astype(bool))

    # Flatten the upper triangle correlation matrices and drop NaNs
    flattened_real = upper_tri_real.values.flatten()
    flattened_synthetic = upper_tri_synthetic.values.flatten()

    flattened_real = flattened_real[~np.isnan(flattened_real)]
    flattened_synthetic = flattened_synthetic[~np.isnan(flattened_synthetic)]

    correlation_between_corrs = np.corrcoef(
        flattened_real, flattened_synthetic)[0, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=flattened_real, y=flattened_synthetic)
    plt.xlabel("Correlations in real data")
    plt.ylabel("Correlations in synthetic data")
    plt.title(
        f"Correlation between correlations, cors:{round(correlation_between_corrs, ndigits=3)}")
    plt.tight_layout()
    plt.savefig(synthetic_data_file.replace(
        ".csv", "_correlation_between_correlations.pdf"))
    plt.show()
    plt.close()

    # Correlation of correlations with response variable
    corr_mmprogression = corr_real["auc_binary"][corr_real["auc_binary"] != 1].corr(
        corr_synthetic["auc_binary"][corr_synthetic["auc_binary"] != 1])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=corr_real["auc_binary"][corr_real["auc_binary"] != 1],
                    y=corr_synthetic["auc_binary"][corr_synthetic["auc_binary"] != 1])
    plt.xlabel("Correlations in real data")
    plt.ylabel("Correlations in synthetic data")
    plt.title(
        f"Correlations of auc_binary in real and synthetic data\nwith other variables, cor of cors:{round(corr_mmprogression, ndigits=3)}")
    plt.tight_layout()
    plt.savefig(synthetic_data_file.replace(
        ".csv", "_correlation_of_auc_binary.pdf"))
    plt.show()
    plt.close()

# %%
