import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np








df = pd.read_csv("results.csv")
df =df[df["ROC_test"]>0.7]
# Assuming hyperparameters are defined
hyperparameters = ['alpha', 'lr_lc', 'gnn_lr', 'batch_size', 'gnn_aggr']

# Determine the number of hyperparameters and calculate the number of rows needed
num_hyperparams = len(hyperparameters)
num_cols = 4
num_rows = np.ceil(num_hyperparams/ num_cols).astype(int)

# Create subplots in a grid of specified columns
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate through it

# Iterate through each hyperparameter
for i, hp in enumerate(hyperparameters):
    # Group by the hyperparameter and get the top 10 ROC results
    top_results = df.groupby(hp).apply(lambda x: x.nlargest(40, 'ROC_test')).reset_index(drop=True)
    
    # Plot the top ROC results for the current hyperparameter
    sns.boxplot(data=top_results, x=hp, y='ROC_test', ax=axes[i], palette="deep")
    # axes[i].plot(top_results[hp], top_results['ROC_test'], marker='o', label=hp)
    axes[i].set_title(f'Top 10 ROC Results for {hp}')
    axes[i].set_xlabel(hp)
    axes[i].set_ylabel('AUC-ROC score')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()

# Hide any remaining empty subplots
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes[j])

plt.tight_layout()



# Group by 'dataset' and sort by ROC and AP scores
grouped = df.groupby('dataset')

# Iterate through each dataset group and find the top 2 ROC and AP scores
for dataset, group in grouped:
    print(f"Dataset: {dataset}")
    
    # Sort by ROC score and get the top 2 rows
    top_roc = group.nlargest(2, 'ROC_test')
    print("\nTop 2 ROC scores:")
    print(top_roc[['ROC_test', 'alpha', 'lr_lc', 'gnn_lr', 'batch_size', 'gnn_aggr', 'epoch_needed', 'epochs', 'patience', 'gnn_num_input', 'gnn_num_output']])
    
    # Sort by AP score and get the top 2 rows
    top_ap = group.nlargest(2, 'AP_test')
    print("\nTop 2 AP scores:")
    print(top_ap[['AP_test', 'alpha', 'lr_lc', 'gnn_lr', 'batch_size', 'gnn_aggr', 'epoch_needed', 'epochs', 'patience', 'gnn_num_input', 'gnn_num_output']])
    
    print("\n" + "-"*50 + "\n")


plt.show()
