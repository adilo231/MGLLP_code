
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from GNNMLP import GNN, Classifier
from pytorch_metric_learning import losses
import pickle

# from lp_datasets import lp_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  average_precision_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import warnings
import os


warnings.filterwarnings('ignore')
plt.style.use('ggplot')



def Get_data_loaders(dataset, num_val=0.1, num_test=0.1, disjoint_train_ratio=0.8, neg_sampling_ratio=1, batch_size=16):
    transformer = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=True)

    train_data, val_data, test_data = transformer(dataset)

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[-1],
        neg_sampling_ratio=0,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[-1],
        neg_sampling_ratio=0,
        edge_label_index=val_data.edge_label_index,
        edge_label=val_data.edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[-1],
        neg_sampling_ratio=0,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, val_loader, test_loader


def Training_Models(train_loader, test_loader, val_loader, GNN_input, batch_size=0, dataset_name='', epochs=60, lr_gnn=0.0001, lr_lc=0.001, display=5, patience=5, alpha=0.1, folder='./results/figs'):
    # model definition

    best_roc = 0

    epochs_no_improve = 0

    GNN_hiddens = 256
    MLP_input = GNN_hiddens//2

    gnn = GNN(input=GNN_input, hidden_channels=GNN_hiddens)
    gnn = gnn.to(device)

    gnn_optim = torch.optim.Adam(gnn.parameters(), lr=lr_gnn)
    gnn_loss = losses.TripletMarginLoss(margin=alpha)
    #gnn_loss = F.cross_entropy

    lc = Classifier(MLP_input*2, 128, 1)
    lc = lc.to(device)

    lc_optim = torch.optim.Adam(lc.parameters(), lr=lr_lc)
    # lc_loss = F.binary_cross_entropy
    lc_loss = nn.BCELoss()

    val_roc = []
    train_roc = []

    train_loss_gnn = []
    train_loss_lc = []

    val_loss_gnn = []
    val_loss_lc = []

    for epoch in tqdm(range(epochs), desc='Epochs', position=0):
        total_loss_gnn = 0
        total_loss_lc = 0

        y_true = []
        y_pred_score = []
        gnn.train()
        lc.train()
        for sampled_data in train_loader:

            gnn_optim.zero_grad()
            lc_optim.zero_grad()

            # Move `sampled_data` to the respective `device`
            sampled_data = sampled_data.to(device)

            # Run `forward` pass of the model
            emb = gnn(sampled_data.x, sampled_data.edge_index)
            
            # Get the ground truth labels from `sampled_data`
            ground_truth = sampled_data.edge_label.to(device)

            # Get link embaddings
            edge_feat_1 = torch.index_select(
                emb, 0, sampled_data.edge_label_index[0])
            edge_feat_2 = torch.index_select(
                emb, 0, sampled_data.edge_label_index[1])

            link_emb = torch.cat([edge_feat_1, edge_feat_2], dim=-1)

            # Get lgnn loss
            loss = gnn_loss(link_emb,
                            ground_truth.to(torch.long))
            
            loss.backward(retain_graph=True)

            # Link classifier
            pred = lc(link_emb)

            loss2 = lc_loss(pred.squeeze(), ground_truth)
            loss2.backward()

            lc_optim.step()
            gnn_optim.step()

            pred_binary = torch.detach(pred).cpu().numpy()
            ground_truth_cpu = ground_truth.cpu().numpy()

            y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
            y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            total_loss_gnn += loss.item()
            total_loss_lc += loss2.item()

        train_roc.append(roc_auc_score(y_true, y_pred_score))
        train_loss_gnn.append(total_loss_gnn/len(train_loader))
        train_loss_lc.append(total_loss_lc/len(train_loader))

        gnn.eval()
        lc.eval()
        with torch.no_grad():
            total_loss_gnn = 0
            total_loss_lc = 0

            y_pred_score = []
            y_true = []
            for sampled_data in val_loader:

                # Move `sampled_data` to the respective `device`
                sampled_data = sampled_data.to(device)

                # Run `forward` pass of the model
                emb = gnn(sampled_data.x, sampled_data.edge_index)

                # Get the ground truth labels from `sampled_data`
                ground_truth = sampled_data.edge_label.to(device)

                edge_feat_1 = torch.index_select(
                    emb, 0, sampled_data.edge_label_index[0])
                edge_feat_2 = torch.index_select(
                    emb, 0, sampled_data.edge_label_index[1])

                link_emb = torch.cat([edge_feat_1, edge_feat_2], dim=-1)

                total_loss_gnn += gnn_loss(link_emb.to(torch.float),
                                           ground_truth.to(torch.long)).item()
                # Link classifier
                pred = lc(link_emb)
                total_loss_lc += lc_loss(pred.squeeze(), ground_truth).item()

                pred_binary = torch.detach(pred).cpu().numpy()
                ground_truth_cpu = ground_truth.cpu().numpy()
                y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
                y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            val_roc.append(roc_auc_score(y_true, y_pred_score))
            val_loss_gnn.append(total_loss_gnn/len(val_loader))
            val_loss_lc.append(total_loss_lc/len(val_loader))

            _, _, _, _, roc, ap = Testing_model(test_loader, gnn, lc)
            if val_roc[-1] > best_roc:
                best_roc = val_roc[-1]

                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # if epoch % display == 0:
            #     tqdm.write(
            #         f"Epoch: {epoch+1}/{epochs} |   Train roc: {train_roc[-1]:.4f}| val roc: {val_roc[-1]:.4f} best roc: {best_roc:.4f} test roc: {roc:.4f} AP {ap:.4f} ")

        # Check if early stopping criteria met
        if epochs_no_improve >= patience:
           
            break

    plt.subplots(1, 3, figsize=(30, 10))
    # Plotting the accuracy curves
    plt.subplot(1, 3, 1)
    plt.plot(train_roc, label='Train Accuracy')
    plt.plot(val_roc, label='Validation Accuracy')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(train_loss_gnn, label='train loss')
    plt.plot(val_loss_gnn, label='vall loss')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(train_loss_lc, label='train loss')
    plt.plot(val_loss_lc, label='val loss')

    # Adding legend
    plt.legend()
    datetime.datetime.now()
    # Saving the figure

    fig_name = f'/TrainVal_plot-{dataset_name}_batch_size{batch_size},alpha_{alpha}_lr:{lr_gnn}-{lr_lc}({ datetime.datetime.now().strftime("%m%d-%H:%M")}).png'
    fig_path = folder + fig_name
    plt.savefig(fig_path)
    acc, pre, recall, f1, roc, ap = Testing_model(test_loader, gnn, lc)
    return acc, pre, recall, f1, roc, ap


def Testing_model(test_loader, gnn, lc):
    y_pred = []
    y_pred_score = []
    y_true = []
    gnn.eval()
    lc.eval()
    for sampled_data in test_loader:

        # Move `sampled_data` to the respective `device`
        sampled_data = sampled_data.to(device)

        # Run `forward` pass of the model
        emb = gnn(sampled_data.x, sampled_data.edge_index)

        # Get the ground truth labels from `sampled_data`
        ground_truth = sampled_data.edge_label.to(device)

        edge_feat_1 = torch.index_select(
            emb, 0, sampled_data.edge_label_index[0])
        edge_feat_2 = torch.index_select(
            emb, 0, sampled_data.edge_label_index[1])

        link_emb = torch.cat([edge_feat_1, edge_feat_2], dim=-1)

        # Link classifier
        pred = lc(link_emb)

        pred_binary = torch.detach(pred).cpu().numpy()
        ground_truth_cpu = ground_truth.cpu().numpy()
        y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])

        y_pred = np.hstack([y_pred, np.round(pred_binary.squeeze())])
        y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            average_precision_score(y_true, y_pred_score),
            roc_auc_score(y_true, y_pred_score))



def Exp_lr(data, dataset_name,df, batch_size=2, alpha=1.0):

    folder_name = f"./results/lr/new-{dataset_name}_{datetime.datetime.now().strftime('%m_%d(%H:%M)')}"
    os.mkdir(folder_name)

    with open(folder_name+'/resultsMeta.txt', 'a') as f:
        f.write(
            f"\n***{datetime.datetime.now().strftime('%Y/%m/%d-(%H:%M)')}****\n")

    
    
    lrs = [0.0001,0.001,0.01]
    for lr_gnn in lrs:
        for lr_lc in lrs:

 
                print(f"lrs: gnn,lc {lr_gnn} {lr_lc}  --> ")
                # print(f"alpha {alpha}  --> ")
                train_loader, val_loader, test_loader = Get_data_loaders(
                    data, batch_size=batch_size)
                # training one model
                # train_loader, val_loader, test_loader = Get_data_loaders(dataset[0], batch_size=8)
                acc, pres, recall, f1, ap, roc = Training_Models(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
                                                                epochs=250, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=20, alpha=alpha, folder=folder_name)
                row_df = pd.DataFrame([[dataset_name,batch_size,alpha,lr_gnn,lr_lc,roc,ap, acc,pres, recall, f1]], columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])

                # Add the row DataFrame to the main DataFrame
                df = pd.concat([df, row_df], ignore_index=True)
                r = f"datasetname={dataset_name},  lr_gnn:{lr_gnn}, lr_lc:{lr_lc} batch_size: {batch_size}, alpha:{alpha} Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"
                with open(folder_name+'/resultsMeta.txt', 'a') as f:
                    f.writelines(r)
                print(r)
    print(df)
    return df
      
    

def Exp_alpha(data, dataset_name,df, batch_size=8, lr_gnn=0.0001, lr_lc=0.0001):
    """
    Perform experimental training and evaluation using different alpha values.

    Args:
        data (torch_geometric.data.Dataset): The dataset for training.
        dataset_name (str): Name of the dataset.
        df (pandas.DataFrame): Existing DataFrame to store results.
        batch_size (int, optional): Batch size for training. Default is 8.
        lr_gnn (float, optional): Learning rate for GNN. Default is 0.0001.
        lr_lc (float, optional): Learning rate for LC. Default is 0.0001.
        alphas (list of float, optional): List of alpha values to try. Default is [1.0].

    Returns:
        df (pandas.DataFrame): Updated DataFrame with appended results.

    """
    folder_name = f"./results/alpha/new-{dataset_name}_{datetime.datetime.now().strftime('%m_%d(%H:%M)')}"
    os.mkdir(folder_name)

    with open(folder_name+'/resultsMeta.txt', 'a') as f:
        f.write(
            f"\n***{datetime.datetime.now().strftime('%Y/%m/%d-(%H:%M)')}****\n")

  
    alphas=[0.001,0.01,0.05,0.1,0.5,1.0,1.5,2.0]
    for alpha in alphas:
            
        print(f"alpha {alpha}  --> ")
        train_loader, val_loader, test_loader = Get_data_loaders(
            data, batch_size=batch_size)
        # training one model
        # train_loader, val_loader, test_loader = Get_data_loaders(dataset[0], batch_size=8)
        acc, pres, recall, f1, ap, roc = Training_Models(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
                                                        epochs=250, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=20, alpha=alpha, folder=folder_name)
        row_df = pd.DataFrame([[dataset_name,batch_size,alpha,lr_gnn,lr_lc,roc,ap, acc,pres, recall, f1]], columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])

        # Add the row DataFrame to the main DataFrame
        df = pd.concat([df, row_df], ignore_index=True)
        r = f"datasetname={dataset_name},  lr_gnn:{lr_gnn}, lr_lc:{lr_lc} batch_size: {batch_size}, alpha:{alpha} Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

        with open(folder_name+'/resultsMeta.txt', 'a') as f:
            f.writelines(r)
            print(r)
       

    return df
      
    

def Exp_batch(data, dataset_name,df, lr_gnn=0.0001, lr_lc=0.0001,alpha=1.0,iters=1):
    """
    Perform experimental training and evaluation using different batch sizes.

    Args:
        data (torch_geometric.data.Dataset): The dataset for training.
        dataset_name (str): Name of the dataset.
        df (pandas.DataFrame): Existing DataFrame to store results.
        lr_gnn (float, optional): Learning rate for GNN. Default is 0.0001.
        lr_lc (float, optional): Learning rate for LC. Default is 0.0001.
        alpha (float, optional): Alpha parameter. Default is 1.0.

    Returns:
        df (pandas.DataFrame): Updated DataFrame with appended results.

    """
    folder_name = f"./results/figs/new-{dataset_name}_{datetime.datetime.now().strftime('%m_%d(%H:%M)')}"
    os.mkdir(folder_name)

    with open(folder_name+'/resultsMeta.txt', 'a') as f:
        f.write(
            f"\n***{datetime.datetime.now().strftime('%Y/%m/%d-(%H:%M)')}****\n")
    
    
    batch_sizes = [128,64,32,16,8,4,2]
    
    for batch_size in batch_sizes:
        for iter in range(iters):    
            print(f"batch size {batch_size}  --> ")
            
            train_loader, val_loader, test_loader = Get_data_loaders(
                data, batch_size=batch_size)
            # training one model
            acc, pres, recall, f1, ap, roc = Training_Models(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
                                                            epochs=150, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=15, alpha=alpha, folder=folder_name)
            row_df = pd.DataFrame([[dataset_name,batch_size,alpha,lr_gnn,lr_lc,roc,ap, acc,pres, recall, f1]], columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])

            # Add the row DataFrame to the main DataFrame
            df = pd.concat([df, row_df], ignore_index=True)
            r = f"datasetname={dataset_name},  lr_gnn:{lr_gnn}, lr_lc:{lr_lc} batch_size: {batch_size}, alpha:{alpha} Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

            with open(folder_name+'/resultsMeta.txt', 'a') as f:
                f.writelines(r)
            print(r)
    print(df)
    return df


def Exp_final(data, dataset_name, batch_size=8, lr_gnn=0.0001, lr_lc=0.001, alpha=1.0):
    """
    Perform experimental training and evaluation using different parameters.

    Args:
        data (torch_geometric.data.Dataset): The dataset for training.
        dataset_name (str): Name of the dataset.
        df (pandas.DataFrame): Existing DataFrame to store results.
        batch_size (int, optional): Batch size for training. Default is 8.
        lr_gnn (float, optional): Learning rate for GNN. Default is 0.0001.
        lr_lc (float, optional): Learning rate for LC. Default is 0.001.
        alpha (float, optional): Alpha parameter. Default is 1.0.

    Returns:
        None.

    """
    # Create a folder to save results
    folder_name = f"./results/figs/finals50/new-{dataset_name}_{datetime.datetime.now().strftime('%m_%d(%H:%M)')}"
    os.mkdir(folder_name)

    # Log metadata to a file
    with open(folder_name+'/resultsMeta.txt', 'a') as f:
        f.write(f"\n***{datetime.datetime.now().strftime('%Y/%m/%d-(%H:%M)')}****\n")
    
    # List of batch sizes to iterate through
    batch_sizes = [2]
    
    # Iterate through different batch sizes
    for batch_size in batch_sizes:
        rocs = []
        aps = []
        
        # Perform experiments for a certain number of iterations
        for iter in range(5):
            print(f"iter {iter+1}  --> ")
            
            # Load data loaders
            train_loader, val_loader, test_loader = Get_data_loaders(
                data, batch_size=batch_size)
            
            # Train models and evaluate
            acc, pres, recall, f1, ap, roc = Training_Models(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
                                                            epochs=150, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=15, alpha=alpha, folder=folder_name)
            
            # Log results to file
            r = f"datasetname={dataset_name},  lr_gnn:{lr_gnn}, lr_lc:{lr_lc} batch_size: {batch_size}, alpha:{alpha} Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"
            rocs.append(roc)
            aps.append(ap)
            with open(folder_name+'/resultsMeta.txt', 'a') as f:
                f.writelines(r)
            print(r)
        
        # Calculate and log aggregated results
        r = f"datasetname={dataset_name},  lr_gnn:{lr_gnn}, lr_lc:{lr_lc} batch_size: {batch_size}, alpha:{alpha} roc: {np.array(rocs).mean()*100:.4f}\xB1{np.array(rocs).std()*100:.4f}, ap: {np.array(aps).mean()*100:.4f}\xB1{np.array(aps).std()*100:.4f}\n\n"
        with open(folder_name+'/resultsMeta.txt', 'a') as f:
                f.writelines(r)
   
      
    


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset_name = 'UAL'
    # if dataset_name == 'Cora':
    #     dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    #     data = dataset[0]
    #     num_features = dataset.num_features
    #     print('Dataset', dataset_name)
    # if dataset_name == 'UAL':
    #     data = lp_dataset(dataset_name)
    #     print('Dataset', dataset_name, len(data.x))


    #     num_features = len(data.x)


    
    # # Exp_lr(data, dataset_name, batch_size=2, alpha=1.0)
    # Exp_alpha(data,dataset_name,batch_size=2,lr_gnn=0.0001,lr_lc=0.0001)
    # # Exp_batch_size(data,dataset_name,lr_gnn=0.0001,lr_lc=0.0001,alpha=1.0 )

    # dataset_name = 'UAL'

    # # check node2vec dim
    # for dimensions in [16,32,64,128,256]:
    #     print('dim --->',dimensions)
    #     pickle_file_path =f"data/datasets/processed/DataTest/{dataset_name}{dimensions}.pkl"
    #     with open(pickle_file_path, "rb") as file:
    #             data = pickle.load( file)
    #     Exp_alpha(data,dataset_name,batch_size=2,lr_gnn=0.0001,lr_lc=0.0001)

    # 'GRQ.pkl', 'BUP.pkl', 'CGS.pkl', 'HMT.pkl', 'NSC.pkl', 
    #            'UPG.pkl', 'HPD.pkl', 'ADV.pkl', 'YST.pkl', 'SMG.pkl',
    # # Test all datasets
    #datasets= [   'FBK.pkl', 'PGP.pkl', 
    #           'UAL.pkl', 'ZWL.pkl', 'ERD.pkl', 'EML.pkl', 'KHN.pkl',
    #           'INF.pkl', 'HTC.pkl', 'CDM.pkl','LDG.pkl',]
               
    # datasets= [   'BUP.pkl','SMG.pkl','EML.pkl','NSC.pkl','YST.pkl','Power.pkl','KHN.pkl','ADV.pkl'
    #                 ,'LDG.pkl','HPD.pkl','GRQ.pkl','ZWL.pkl',]
    
    # datasets = [ 'KHN.pkl',]
    # batch_sizes = [64,32,16,8,4,2]
    # lrs=[0.1,0.01,0.001,0.0001]

    # df = pd.DataFrame(columns=["batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])
    # print(datasets)
    # for dataset_name in datasets:
    #         print(dataset_name)
    #         pickle_file_path =f"data/datasets/processed/{dataset_name}"
    #         with open(pickle_file_path, "rb") as file:
    #                 data = pickle.load( file)
    #         for batch_size in batch_sizes:
    #             for lr_gnn in lrs:
    #                 for lr_lc in lrs:
                        
    #                     df =Exp_alpha(data,dataset_name.split('.')[0],df=df,batch_size=batch_size,lr_gnn=lr_gnn,lr_lc=lr_lc)
    #                     df.to_csv("results/meta.csv")
    # 'SMG.pkl','EML.pkl', 'NSC.pkl','YST.pkl',
                        
    # datasets= [ 'KHN.pkl','ADV.pkl'
    #                 ,'LDG.pkl','HPD.pkl','GRQ.pkl','BUP.pkl','ZWL.pkl', ]
    
    

    # try:
    #     df = pd.read_csv("results/meta_alpha.csv",index_col=0)
    # except:
    #     df = pd.DataFrame(columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])
    # df_dataset = pd.read_csv("results/datasets.csv",index_col=0)
    # # Iterate through the sorted DataFrame and print the "datasetname" column
    # for index, row in df_dataset.iterrows():
    #         dataset_name=row["dataset"]
    #         print(dataset_name)
    #         pickle_file_path =f"data/datasets/processed/{dataset_name}"
    #         with open(pickle_file_path, "rb") as file:
    #                 data = pickle.load( file)

                        
    #         df =Exp_alpha(data,dataset_name.split('.')[0]+"_alpha",df=df,batch_size=8,lr_gnn=0.0001,lr_lc=0.001)
    #         df.to_csv("results/meta_alpha.csv")
    
    


    
    try:
        df = pd.read_csv("results/meta_batch.csv",index_col=0)
    except:
        df = pd.DataFrame(columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])
 


    
    df_dataset = pd.read_csv("results/datasets.csv",index_col=0)
    # Iterate through the sorted DataFrame and print the "datasetname" column
    for index, row in df_dataset.iterrows():
            dataset_name=row["dataset"]
    # for dataset_name in datasets:
            print(dataset_name)
            pickle_file_path =f"data/datasets/processed/{dataset_name}"
            with open(pickle_file_path, "rb") as file:
                    data = pickle.load( file)
        
                        
            df =Exp_batch(data,dataset_name.split('.')[0]+"_batch",df=df,lr_gnn=0.0001,lr_lc=0.0001,alpha=1.0)
            df.to_csv("results/meta_batch.csv")



    # try:
    #     df = pd.read_csv("results/meta_lr.csv",index_col=0)
    # except:
    #     df = pd.DataFrame(columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])
 


  
    
    # df_dataset = pd.read_csv("results/datasets.csv",index_col=0)
    # # Iterate through the sorted DataFrame and print the "datasetname" column
    # for index, row in df_dataset.iterrows():
    #         dataset_name=row["dataset"]
    #         print(dataset_name)
    #         pickle_file_path =f"data/datasets/processed/{dataset_name}"
    #         with open(pickle_file_path, "rb") as file:
    #                 data = pickle.load( file)
                        
    #         df =Exp_lr(data,dataset_name.split('.')[0]+"_lrs",df=df,batch_size=8,alpha=1.0)
    #         df.to_csv("results/meta_lr.csv")



    # df = pd.DataFrame(columns=["dataset","batchsize","alpha","gnnlr","lclr","roc","ap", "acc",'pres', "recall", "f1"])
 


    
    # df_dataset = pd.read_csv("results/datasets.csv",index_col=0)
    # # Iterate through the sorted DataFrame and print the "datasetname" column
    # for index, row in df_dataset.iterrows():
    #         dataset_name=row["dataset"]
    # # for dataset_name in datasets:
    #         print(dataset_name)
    #         if index >= 0:
    #             pickle_file_path =f"data/datasets/processed/{dataset_name}"
    #             with open(pickle_file_path, "rb") as file:
    #                     data = pickle.load( file)
            
                            
    #             Exp_final(data,dataset_name.split('.')[0]+"_final",batch_size=4,lr_gnn=0.0001,lr_lc=0.001)
                
