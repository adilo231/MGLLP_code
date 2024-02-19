import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# df = pd.read_csv("results/meta.csv",index_col=0)
# aux = df[["gnnlr","lclr","roc"]]
# print(aux)
# aux =aux.pivot("gnnlr","lclr","roc")

# sns.heatmap(aux, annot=True,fmt=".2f")

# plt.show()

# datasets= [ 'SMG.pkl',
#             'EML.pkl',
#             'NSC.pkl',
#             'YST.pkl',
#             'KHN.pkl',
#             'ADV.pkl',
#             'LDG.pkl',
#             'HPD.pkl',
#             'GRQ.pkl',
#             'BUP.pkl',
#             'ZWL.pkl',
#             'UPG.pkl', 
#             'CEG.pkl',
#             'UAL.pkl'
#             ]
# num_nodes=[1024,
#            1133,
#            1461,
#            2284,
#            3772,
#            5155,
#            8324,
#            8756,
#            5241,
#            105,
#            6651,
#            4941,
#            297,
#            332
#             ]









# df = pd.DataFrame(columns=["dataset","num_nodes","batchsize","alpha","gnnlr","lclr"])
# for dataset_name,nodes in zip(datasets,num_nodes):
#     row_df = pd.DataFrame([[dataset_name,nodes,2,1.0,0.0001,0.001]], columns=["dataset","num_nodes","batchsize","alpha","gnnlr","lclr"])

#     df = pd.concat([df, row_df], ignore_index=True)
# df = df.sort_values(by="num_nodes")

# print(df)
# df.to_csv("results/datasets.csv")
# # Iterate through the sorted DataFrame and print the "datasetname" column



# df = pd.read_csv("results/datasets.csv",index_col=0)

# for index, row in df.iterrows():
#     print(row["dataset"])

print("Best values for alpha \n ***********************")
dfalpha = pd.read_csv("results/meta_alpha.csv",index_col=0)

print(dfalpha.loc[dfalpha.groupby("dataset")["roc"].idxmax(),[ "dataset","alpha","roc"]],dfalpha.groupby("dataset")['roc'].std()*100 )

print(dfalpha.groupby("alpha")["roc"].mean()*100,dfalpha.groupby("alpha")["roc"].std()*100)



# Get unique dataset names
unique_datasets = dfalpha["dataset"].unique()

# Create subplots
fig, axes = plt.subplots(nrows=len(unique_datasets), ncols=1, figsize=(8, 2 * len(unique_datasets)))


['YTS','BUP','ADV']
for i, dataset_name in enumerate(unique_datasets):
    ax = axes[i]
    dataset_df = dfalpha[dfalpha["dataset"] == dataset_name]
    
    ax.plot(dataset_df["alpha"], dataset_df["roc"], marker='o', label=dataset_name)
    ax.legend()
    
    
    # Remove the following line to keep x-axis tick labels at the bottom-most subplot
    if i != len(unique_datasets) - 1:
        ax.set_xticks([])
  
   
    

plt.tight_layout()










print("Best values for bqtch size \n ***********************")
dfalpha = pd.read_csv("results/meta_batch.csv",index_col=0)

print(dfalpha.loc[dfalpha.groupby("dataset")["roc"].idxmax(),[ "dataset","batchsize"]] )

print(dfalpha.groupby("batchsize")["roc"].mean())


plt.figure()
dfalpha.groupby("batchsize")["roc"].mean().plot()

print("Best values for lr  \n ***********************")
dfalpha = pd.read_csv("results/meta_lr.csv",index_col=0)

print(dfalpha.loc[dfalpha.groupby("dataset")["roc"].idxmax(),[ "dataset","gnnlr","lclr",]] )

print(dfalpha.groupby(["gnnlr","lclr"])["roc"].mean()*100,dfalpha.groupby(["gnnlr","lclr"])["roc"].std()*100)


