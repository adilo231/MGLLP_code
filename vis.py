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

Epoh: 1, gnn Loss: 0.2391, lc Loss 0.7291, 
Epoh: 2, gnn Loss: 0.0663, lc Loss 0.6946, 
Epoh: 3, gnn Loss: 0.0155, lc Loss 0.6937, 
Epoh: 4, gnn Loss: 0.0117, lc Loss 0.6939, 
Epoh: 5, gnn Loss: 0.0099, lc Loss 0.6937, 
Epoh: 6, gnn Loss: 0.0079, lc Loss 0.6937, 
Epoh: 7, gnn Loss: 0.0087, lc Loss 0.6938, 
Epoh: 8, gnn Loss: 0.0064, lc Loss 0.6945, 
Epoh: 9, gnn Loss: 0.0074, lc Loss 0.6941, 
Epoc: 10, gnn Loss: 0.0063, lc Loss 0.6938, 
Epoc: 11, gnn Loss: 0.0064, lc Loss 0.6941, 
Epoc: 12, gnn Loss: 0.0049, lc Loss 0.6941, 
Epoc: 13, gnn Loss: 0.0049, lc Loss 0.6939, 
Epoc: 14, gnn Loss: 0.0049, lc Loss 0.6937, 
Epoc: 15, gnn Loss: 0.0046, lc Loss 0.6946, 
Epoc: 16, gnn Loss: 0.0042, lc Loss 0.6937, 
Epoc: 17, gnn Loss: 0.0051, lc Loss 0.6943, 
Epoc: 18, gnn Loss: 0.0045, lc Loss 0.6937, 
Epoc: 19, gnn Loss: 0.0045, lc Loss 0.6949, 
Epoc: 20, gnn Loss: 0.0042, lc Loss 0.6943, 
Epoc: 21, gnn Loss: 0.0034, lc Loss 0.6942, 
Epoc: 22, gnn Loss: 0.0037, lc Loss 0.6948, 
Epoc: 23, gnn Loss: 0.0034, lc Loss 0.6936, 
Epoc: 24, gnn Loss: 0.0026, lc Loss 0.6939, 
Epoc: 25, gnn Loss: 0.0026, lc Loss 0.6953, 
Epoc: 26, gnn Loss: 0.0024, lc Loss 0.6937, 
Epoc: 27, gnn Loss: 0.0037, lc Loss 0.6944, 
Epoc: 28, gnn Loss: 0.0037, lc Loss 0.6941, 
Epoc: 29, gnn Loss: 0.0039, lc Loss 0.6947, 
Epoc: 30, gnn Loss: 0.0031, lc Loss 0.6944, 
Epoc: 31, gnn Loss: 0.0024, lc Loss 0.6944, 
Epoc: 32, gnn Loss: 0.0023, lc Loss 0.6939, 
Epoc: 33, gnn Loss: 0.0026, lc Loss 0.6937, 
Epoc: 34, gnn Loss: 0.0034, lc Loss 0.6940, 
Epoc: 35, gnn Loss: 0.0023, lc Loss 0.6939, 
Epoc: 36, gnn Loss: 0.0021, lc Loss 0.6935, 
Epoc: 37, gnn Loss: 0.0025, lc Loss 0.6934, 
Epoc: 38, gnn Loss: 0.0026, lc Loss 0.6943, 
Epoc: 39, gnn Loss: 0.0028, lc Loss 0.6937, 
Epoc: 40, gnn Loss: 0.0022, lc Loss 0.6938, 
Epoc: 41, gnn Loss: 0.0027, lc Loss 0.6934, 
Epoc: 42, gnn Loss: 0.0023, lc Loss 0.6947, 
Epoc: 43, gnn Loss: 0.0023, lc Loss 0.6936, 
Epoc: 44, gnn Loss: 0.0023, lc Loss 0.6938, 
Epoc: 45, gnn Loss: 0.0026, lc Loss 0.6933, 
Epoc: 46, gnn Loss: 0.0017, lc Loss 0.6936, 
Epoc: 47, gnn Loss: 0.0017, lc Loss 0.6933, 
Epoc: 48, gnn Loss: 0.0027, lc Loss 0.6935, 
Epoc: 49, gnn Loss: 0.0024, lc Loss 0.6942, 
Epoc: 50, gnn Loss: 0.0025, lc Loss 0.6933, 
Epoc: 51, gnn Loss: 0.0020, lc Loss 0.6934, 
Epoc: 52, gnn Loss: 0.0023, lc Loss 0.6938, 
Epoc: 53, gnn Loss: 0.0025, lc Loss 0.6932, 
Epoc: 54, gnn Loss: 0.0014, lc Loss 0.6933, 
Epoc: 55, gnn Loss: 0.0015, lc Loss 0.6931, 
Epoc: 56, gnn Loss: 0.0019, lc Loss 0.6930, 
Epoc: 57, gnn Loss: 0.0011, lc Loss 0.6930, 
Epoc: 58, gnn Loss: 0.0014, lc Loss 0.6929, 
Epoc: 59, gnn Loss: 0.0014, lc Loss 0.6931, 
Epoc: 60, gnn Loss: 0.0013, lc Loss 0.6929, 
Epoc: 61, gnn Loss: 0.0009, lc Loss 0.6926, 
Epoc: 62, gnn Loss: 0.0010, lc Loss 0.6923, 
Epoc: 63, gnn Loss: 0.0010, lc Loss 0.6926, 
Epoc: 64, gnn Loss: 0.0005, lc Loss 0.6924, 
Epoc: 65, gnn Loss: 0.0006, lc Loss 0.6923, 
Epoc: 66, gnn Loss: 0.0009, lc Loss 0.6927, 
Epoc: 67, gnn Loss: 0.0004, lc Loss 0.6918, 
Epoc: 68, gnn Loss: 0.0005, lc Loss 0.6918, 
Epoc: 69, gnn Loss: 0.0006, lc Loss 0.6910, 
Epoc: 70, gnn Loss: 0.0003, lc Loss 0.6914, 
Epoc: 71, gnn Loss: 0.0003, lc Loss 0.6904, 
Epoc: 72, gnn Loss: 0.0004, lc Loss 0.6903, 
Epoc: 73, gnn Loss: 0.0005, lc Loss 0.6904, 
Epoc: 74, gnn Loss: 0.0003, lc Loss 0.6900, 
Epoc: 75, gnn Loss: 0.0003, lc Loss 0.6905, 
Epoc: 76, gnn Loss: 0.0001, lc Loss 0.6913, 
Epoc: 77, gnn Loss: 0.0001, lc Loss 0.6888, 
Epoc: 78, gnn Loss: 0.0003, lc Loss 0.6874, 
Epoc: 79, gnn Loss: 0.0003, lc Loss 0.6880, 
Epoc: 80, gnn Loss: 0.0002, lc Loss 0.6864, 
Epoc: 81, gnn Loss: 0.0003, lc Loss 0.6857, 
Epoc: 82, gnn Loss: 0.0002, lc Loss 0.6842, 
Epoc: 83, gnn Loss: 0.0002, lc Loss 0.6825, 
Epoc: 84, gnn Loss: 0.0002, lc Loss 0.6800, 
Epoc: 85, gnn Loss: 0.0002, lc Loss 0.6761, 
Epoc: 86, gnn Loss: 0.0002, lc Loss 0.6743, 
Epoc: 87, gnn Loss: 0.0001, lc Loss 0.6699, 
Epoc: 88, gnn Loss: 0.0001, lc Loss 0.6630, 
Epoc: 89, gnn Loss: 0.0000, lc Loss 0.6547, 
Epoc: 90, gnn Loss: 0.0001, lc Loss 0.6465, 
Epoc: 91, gnn Loss: 0.0001, lc Loss 0.6349, 
Epoc: 92, gnn Loss: 0.0000, lc Loss 0.6201, 
Epoc: 93, gnn Loss: 0.0002, lc Loss 0.6010, 
Epoc: 94, gnn Loss: 0.0000, lc Loss 0.5933, 
Epoc: 95, gnn Loss: 0.0001, lc Loss 0.5730, 
Epoc: 96, gnn Loss: 0.0000, lc Loss 0.5654, 
Epoc: 97, gnn Loss: 0.0000, lc Loss 0.5503, 
Epoc: 98, gnn Loss: 0.0000, lc Loss 0.5477, 
Epoc: 99, gnn Loss: 0.0000, lc Loss 0.5372, 
Epoch 100, gnn Loss: 0.0001, lc Loss 0.5346, 
Epoch 101, gnn Loss: 0.0000, lc Loss 0.5201, 
Epoch 102, gnn Loss: 0.0001, lc Loss 0.5138, 
Epoch 103, gnn Loss: 0.0000, lc Loss 0.5080, 
Epoch 104, gnn Loss: 0.0000, lc Loss 0.5040, 
Epoch 105, gnn Loss: 0.0000, lc Loss 0.4997, 
Epoch 106, gnn Loss: 0.0000, lc Loss 0.4943, 
Epoch 107, gnn Loss: 0.0000, lc Loss 0.4876, 
Epoch 108, gnn Loss: 0.0000, lc Loss 0.4805, 
Epoch 109, gnn Loss: 0.0000, lc Loss 0.4702, 
Epoch 110, gnn Loss: 0.0000, lc Loss 0.4681, 
Epoch 111, gnn Loss: 0.0000, lc Loss 0.4587, 
Epoch 112, gnn Loss: 0.0000, lc Loss 0.4518, 
Epoch 113, gnn Loss: 0.0000, lc Loss 0.4433, 
Epoch 114, gnn Loss: 0.0000, lc Loss 0.4391, 
Epoch 115, gnn Loss: 0.0000, lc Loss 0.4371, 
Epoch 116, gnn Loss: 0.0000, lc Loss 0.4257, 
Epoch 117, gnn Loss: 0.0001, lc Loss 0.4181, 
Epoch 118, gnn Loss: 0.0000, lc Loss 0.4147, 
Epoch 119, gnn Loss: 0.0001, lc Loss 0.4105, 
Epoch 120, gnn Loss: 0.0000, lc Loss 0.3993, 
Epoch 121, gnn Loss: 0.0000, lc Loss 0.3932, 
Epoch 122, gnn Loss: 0.0000, lc Loss 0.3856, 
Epoch 123, gnn Loss: 0.0000, lc Loss 0.3809, 
Epoch 124, gnn Loss: 0.0000, lc Loss 0.3724, 
Epoch 125, gnn Loss: 0.0000, lc Loss 0.3699, 
Epoch 126, gnn Loss: 0.0000, lc Loss 0.3603, 
Epoch 127, gnn Loss: 0.0000, lc Loss 0.3547, 
Epoch 128, gnn Loss: 0.0000, lc Loss 0.3464, 
Epoch 129, gnn Loss: 0.0000, lc Loss 0.3470, 
Epoch 130, gnn Loss: 0.0000, lc Loss 0.3324, 
Epoch 131, gnn Loss: 0.0000, lc Loss 0.3254, 
Epoch 132, gnn Loss: 0.0000, lc Loss 0.3269, 
Epoch 133, gnn Loss: 0.0000, lc Loss 0.3165, 
Epoch 134, gnn Loss: 0.0000, lc Loss 0.3057, 
Epoch 135, gnn Loss: 0.0000, lc Loss 0.3058, 
Epoch 136, gnn Loss: 0.0000, lc Loss 0.3010, 
Epoch 137, gnn Loss: 0.0000, lc Loss 0.2897, 
Epoch 138, gnn Loss: 0.0001, lc Loss 0.2843, 
Epoch 139, gnn Loss: 0.0000, lc Loss 0.2695, 
Epoch 140, gnn Loss: 0.0000, lc Loss 0.2690, 
Epoch 141, gnn Loss: 0.0000, lc Loss 0.2635, 
Epoch 142, gnn Loss: 0.0001, lc Loss 0.2610, 
Epoch 143, gnn Loss: 0.0000, lc Loss 0.2485, 
Epoch 144, gnn Loss: 0.0000, lc Loss 0.2434, 
Epoch 145, gnn Loss: 0.0000, lc Loss 0.2356, 
Epoch 146, gnn Loss: 0.0000, lc Loss 0.2323, 
Epoch 147, gnn Loss: 0.0000, lc Loss 0.2242, 
Epoch 148, gnn Loss: 0.0000, lc Loss 0.2169, 
Epoch 149, gnn Loss: 0.0000, lc Loss 0.2194, 
Epoch 150, gnn Loss: 0.0000, lc Loss 0.2058, 
Epoch 151, gnn Loss: 0.0000, lc Loss 0.2024, 
Epoch 152, gnn Loss: 0.0000, lc Loss 0.1971, 
Epoch 153, gnn Loss: 0.0000, lc Loss 0.1879, 
Epoch 154, gnn Loss: 0.0000, lc Loss 0.1826, 
Epoch 155, gnn Loss: 0.0000, lc Loss 0.1797, 
Epoch 156, gnn Loss: 0.0000, lc Loss 0.1762, 
Epoch 157, gnn Loss: 0.0000, lc Loss 0.1721, 
Epoch 158, gnn Loss: 0.0000, lc Loss 0.1615, 
Epoch 159, gnn Loss: 0.0000, lc Loss 0.1577, 
Epoch 160, gnn Loss: 0.0000, lc Loss 0.1507, 
Epoch 161, gnn Loss: 0.0000, lc Loss 0.1440, 
Epoch 162, gnn Loss: 0.0000, lc Loss 0.1419, 
Epoch 163, gnn Loss: 0.0000, lc Loss 0.1359, 
Epoch 164, gnn Loss: 0.0000, lc Loss 0.1333, 
Epoch 165, gnn Loss: 0.0000, lc Loss 0.1290, 
Epoch 166, gnn Loss: 0.0000, lc Loss 0.1245, 
Epoch 167, gnn Loss: 0.0000, lc Loss 0.1177, 
Epoch 168, gnn Loss: 0.0000, lc Loss 0.1150, 
Epoch 169, gnn Loss: 0.0000, lc Loss 0.1111, 
Epoch 170, gnn Loss: 0.0000, lc Loss 0.1090, 
Epoch 171, gnn Loss: 0.0000, lc Loss 0.1036, 
Epoch 172, gnn Loss: 0.0000, lc Loss 0.0972, 
Epoch 173, gnn Loss: 0.0000, lc Loss 0.0962, 
Epoch 174, gnn Loss: 0.0000, lc Loss 0.0933, 
Epoch 175, gnn Loss: 0.0000, lc Loss 0.0887, 
Epoch 176, gnn Loss: 0.0000, lc Loss 0.0859, 
Epoch 177, gnn Loss: 0.0000, lc Loss 0.0812, 
Epoch 178, gnn Loss: 0.0000, lc Loss 0.0783, 
Epoch 179, gnn Loss: 0.0000, lc Loss 0.0776, 
Epoch 180, gnn Loss: 0.0000, lc Loss 0.0781, 
Epoch 181, gnn Loss: 0.0000, lc Loss 0.0742, 
Epoch 182, gnn Loss: 0.0000, lc Loss 0.0698, 
Epoch 183, gnn Loss: 0.0000, lc Loss 0.0656, 
Epoch 184, gnn Loss: 0.0000, lc Loss 0.0648, 
Epoch 185, gnn Loss: 0.0000, lc Loss 0.0608, 
Epoch 186, gnn Loss: 0.0000, lc Loss 0.0601, 
Epoch 187, gnn Loss: 0.0000, lc Loss 0.0584, 
Epoch 188, gnn Loss: 0.0000, lc Loss 0.0569, 
Epoch 189, gnn Loss: 0.0000, lc Loss 0.0542, 
Epoch 190, gnn Loss: 0.0000, lc Loss 0.0522, 
Epoch 191, gnn Loss: 0.0000, lc Loss 0.0504, 
Epoch 192, gnn Loss: 0.0000, lc Loss 0.0489, 
Epoch 193, gnn Loss: 0.0000, lc Loss 0.0481, 
Epoch 194, gnn Loss: 0.0000, lc Loss 0.0489, 
Epoch 195, gnn Loss: 0.0000, lc Loss 0.0538, 
Epoch 196, gnn Loss: 0.0000, lc Loss 0.0435, 
Epoch 197, gnn Loss: 0.0000, lc Loss 0.0423, 
Epoch 198, gnn Loss: 0.0000, lc Loss 0.0402, 
Epoch 199, gnn Loss: 0.0000, lc Loss 0.0397, 
Epoch 200, gnn Loss: 0.0000, lc Loss 0.0383, 
Epoch 201, gnn Loss: 0.0000, lc Loss 0.0386, 
Epoch 202, gnn Loss: 0.0000, lc Loss 0.0365, 
Epoch 203, gnn Loss: 0.0000, lc Loss 0.0340, 
Epoch 204, gnn Loss: 0.0000, lc Loss 0.0349, 
Epoch 205, gnn Loss: 0.0000, lc Loss 0.0336, 
Epoch 206, gnn Loss: 0.0000, lc Loss 0.0321, 
Epoch 207, gnn Loss: 0.0000, lc Loss 0.0308, 
Epoch 208, gnn Loss: 0.0000, lc Loss 0.0297, 
Epoch 209, gnn Loss: 0.0000, lc Loss 0.0284, 
Epoch 210, gnn Loss: 0.0000, lc Loss 0.0274, 
Epoch 211, gnn Loss: 0.0000, lc Loss 0.0271, 
Epoch 212, gnn Loss: 0.0000, lc Loss 0.0268, 
Epoch 213, gnn Loss: 0.0000, lc Loss 0.0253, 
Epoch 214, gnn Loss: 0.0000, lc Loss 0.0249, 
Epoch 215, gnn Loss: 0.0000, lc Loss 0.0244, 
Epoch 216, gnn Loss: 0.0000, lc Loss 0.0237, 
Epoch 217, gnn Loss: 0.0000, lc Loss 0.0225, 
Epoch 218, gnn Loss: 0.0000, lc Loss 0.0229, 
Epoch 219, gnn Loss: 0.0000, lc Loss 0.0218, 
Epoch 220, gnn Loss: 0.0000, lc Loss 0.0212, 
Epoch 221, gnn Loss: 0.0000, lc Loss 0.0205, 
Early stopping triggered after 221 epochs.
Test ROC AUC: 0.9426, Test AP: 0.9366
