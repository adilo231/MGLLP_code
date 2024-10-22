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
num_rows = np.ceil(num_hyperparams*2 / num_cols).astype(int)

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
    axes[i].set_ylabel('ROC')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()

# Hide any remaining empty subplots
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming you already have your DataFrame 'df' and hyperparameters defined
# df=df2.copy()
# # Prepare a dictionary to hold the average top 5 ROC results for each hyperparameter
# average_top_5_roc = {}

# # Iterate through each hyperparameter
# for hp in hyperparameters:
#     # Group by the hyperparameter and get the top 5 ROC results
#     top_results = df.groupby(hp).apply(lambda x: x.nlargest(5, 'ROC_test')).reset_index(drop=True)
    
#     # Calculate the average of the top 5 ROC results for each hyperparameter value
#     average_roc = top_results.groupby(hp)['ROC_test'].mean()
    
#     # Store the average ROC results in the dictionary
#     average_top_5_roc[hp] = average_roc

# # Create a DataFrame from the average results for easy plotting
# average_df = pd.DataFrame(average_top_5_roc)

# # Convert index to string if necessary
# average_df.index = average_df.index.astype(str)

# # Plotting
# plt.figure(figsize=(12, 6))
# for hp in hyperparameters:
#     plt.plot(average_df.index, average_df[hp], marker='o', label=hp)

# plt.title('Average Top 5 ROC Results for Each Hyperparameter')
# plt.xlabel('Hyperparameter Values')
# plt.ylabel('Average ROC')
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

plt.show()
