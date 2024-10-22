import subprocess
import os

# Open output and error files once
out_file = open("training_output/rej_baseline_out.txt", "a")
err_file = open("training_output/rej_baseline_err.txt", "a")

# Define the hyperparameter values you want to vary
datasets = ["BUP","ADV",'EML',"CEG","NSC","SMG"]  # Add more datasets if needed
alpha_values = [0.5, 1.0, 1.5]
gnn_lr_values = [ 0.001, 0.0001]
lc_lr_values = [0.000001, 0.00001]
batch_sizes = [256, 128, 64]
aggrs=['max','min','mean']
# Directory to store logs and results
output_dir = "training_output"
os.makedirs(output_dir, exist_ok=True)

print("Starting experiments")

# Default values for hyperparameters (when not varied)
default_batch_size = 128
default_alpha = 1.0
default_gnn_lr = 0.001
default_lc_lr = 0.000001

# Test each hyperparameter independently
for dataset in datasets:
    for aggr in aggrs:
    # Vary batch size
    # for batch_size in batch_sizes:
    #     print(f"Starting batch_size={batch_size}")
    #     try:
    #         cmd = [
    #             "python", "LineML.py",
    #             f"--dataset={datasets[0]}",
    #             f"--batch_size={batch_size}",
    #             f'--gnn_aggr={aggr}',
    #             f"--alpha={default_alpha}",
    #             f"--gnn_lr={default_gnn_lr}",
    #             f"--lr_lc={default_lc_lr}"
    #         ]
            
    #         subprocess.run(cmd, stdout=out_file, stderr=err_file, check=True)
    #         print(f"Completed batch_size={batch_size}")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error executing with batch_size={batch_size}: {e}")

        for batch_size in batch_sizes:
            for alpha in alpha_values:
                print(f"Starting alpha={alpha}")
                try:
                    cmd = [
                        "python", "LineML.py",
                        f"--dataset={dataset}",
                        f'--gnn_aggr={aggr}',
                        f"--batch_size={batch_size}",
                        f"--alpha={alpha}",
                        f"--gnn_lr={default_gnn_lr}",
                        f"--lr_lc={default_lc_lr}"
                    ]
                    print(cmd)
                    subprocess.run(cmd, stdout=out_file, stderr=err_file, check=True)
                    print(f"Completed alpha={alpha}")
                except subprocess.CalledProcessError as e:
                    print(f"Error executing with alpha={alpha}: {e}")

        for batch_size in batch_sizes:
            for gnn_lr in gnn_lr_values:
                for lc_lr in lc_lr_values:
                    print(f"Starting lc_lr={lc_lr} gnn_lr={gnn_lr}")
                    try:
                        cmd = [
                            "python", "LineML.py",
                            f"--dataset={dataset}",
                            f'--gnn_aggr={aggr}',
                            f"--batch_size={batch_size}",
                            f"--alpha={default_alpha}",
                            f"--gnn_lr={gnn_lr}",
                            f"--lr_lc={lc_lr}"
                        ]
                        
                        subprocess.run(cmd, stdout=out_file, stderr=err_file, check=True)
                        print(f"Completed lc_lr={lc_lr}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing with lc_lr={lc_lr}: {e}")

# Close the output and error files
out_file.close()
err_file.close()



