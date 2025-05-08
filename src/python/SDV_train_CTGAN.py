import matplotlib.pyplot as plt
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd
import numpy as np
import torch
import random
import os

# Path to data
data_path = "/home/nestorask/Projects/Machine_learning_tutorial_HBio2025_server/data/real/"

# Folder to save outputs
path2save_models = "/home/nestorask/Projects/Machine_learning_tutorial_HBio2025_server/results/CTGAN_models/"
path2save_synthdata = "/home/nestorask/Projects/Machine_learning_tutorial_HBio2025_server/data/synthetic/"

# Select train files
train_files = [data_path+f for f in os.listdir(
    data_path) if f.endswith('_training_data.csv')]

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Generate synthetic data
counter = 1
for train_filei in train_files:
    print(f"Generating synthetic data {counter} out of {len(train_files)}")

    # Read real data
    real_data = pd.read_csv(train_filei)
    real_data.drop(columns=["labid"], inplace=True)

    real_data = real_data.iloc[:, 0:51]

    # Make metadata
    metadata = Metadata.detect_from_dataframe(data=real_data)
    metadata.update_column(column_name="auc_binary", sdtype="categorical")

    # Initialize and fit the CTGAN synthesizer
    synthesizer = CTGANSynthesizer(
        metadata, epochs=5000, batch_size=100, verbose=True)
    synthesizer.fit(real_data)

    synthesizer_file = f"{path2save_models}Synthesizer_{
        os.path.basename(train_filei).replace(".csv", ".pkl")}"
    synthesizer.save(filepath=synthesizer_file)

    # Plot Generator and Discriminator losses in relation to Epoch
    losses = synthesizer.get_loss_values()
    plt.figure(figsize=(10, 5))
    plt.plot(losses['Generator Loss'], label='Generator Loss')
    plt.plot(losses['Discriminator Loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses over Epochs')
    plt.legend()
    plt.savefig(synthesizer_file.replace(".pkl", ".pdf"))

    # Create synthetic data
    synthetic_data = synthesizer.sample(num_rows=1000)
    synthetic_data.to_csv(synthesizer_file.replace(
        ".pkl", ".csv"), index=False)

    counter = counter + 1

print("All done")
