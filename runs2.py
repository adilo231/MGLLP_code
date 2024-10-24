import subprocess
import os
import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run GNN experiments with varying hyperparameters.")
parser.add_argument("--device", type=int, choices=[0, 1], required=True, help="Specify the device number (0 or 1).")
args = parser.parse_args()

# Read the dataset from a CSV file
df = pd.read_csv('path/to/your/data.csv')

# Open output and error files once
out_file = open("training_output/rej_baseline_out.txt", "a")
err_file = open("training_output/rej_baseline_err.txt", "a")

# Define the specific datasets you want to run experiments for
datasets_to_run = ["EML", "BUP"]  # Add more datasets as needed

# Function to run the experiment 5 times
def run_experiment(cmd):
    for _ in range(5):
        try:
            subprocess.run(cmd, stdout=out_file, stderr=err_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing: {e}")

# Iterate over each dataset in the datasets_to_run array
for dataset in datasets_to_run:
    # Filter for the current dataset
    dataset_df = df[df['dataset'] == dataset]
    
    # Select the row with the best ROC_test score
    best_row = dataset_df.loc[dataset_df['ROC_test'].idxmax()]
    best_alpha = best_row['alpha']
    best_gnn_lr = best_row['gnn_lr']
    best_lc_lr = best_row['lr_lc']
    best_batch_size = best_row['batch_size']
    best_aggr = best_row['gnn_aggr']

    print(f"Running with best hyperparameters for {dataset}: alpha={best_alpha}, gnn_lr={best_gnn_lr}, batch_size={best_batch_size}")

    # Prepare the command
    cmd = [
        "python", "LineML.py",
        f"--dataset={dataset}",
        f"--batch_size={best_batch_size}",
        f"--alpha={best_alpha}",
        f"--gnn_lr={best_gnn_lr}",
        f"--lr_lc={best_lc_lr}",
        f"--gnn_aggr={best_aggr}",
        f"--device={args.device}"  # Add device argument from command line
    ]

    print("Command:", cmd)
    
    # Run the experiment 5 times
    run_experiment(cmd)

# Close the output and error files
out_file.close()
err_file.close()
