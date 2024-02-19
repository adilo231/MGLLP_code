
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from GNNMLP import GNN,GNN1, Classifier,Classifier1
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



def Get_data_loaders(dataset, num_val=0.1, num_test=0.1, disjoint_train_ratio=0.2, neg_sampling_ratio=1, batch_size=16):
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



def Training_Models_gcn_lc(train_loader, test_loader, val_loader, GNN_input, batch_size=0, dataset_name='', epochs=60, lr_gnn=0.0001, lr_lc=0.001, display=5, patience=5, alpha=0.1, folder='./results/figs'):
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

def Training_Models_gcn(train_loader, test_loader, val_loader, GNN_input, batch_size=0, dataset_name='', epochs=60, lr_gnn=0.0001, lr_lc=0.001, display=5, patience=5, alpha=0.1, folder='./results/figs'):
    # model definition

    best_roc = 0

    epochs_no_improve = 0

    GNN_hiddens = 256
    MLP_input = GNN_hiddens//2

    gnn = GNN1(input=GNN_input, hidden_channels=GNN_hiddens)
    gnn = gnn.to(device)

    gnn_optim = torch.optim.Adam(gnn.parameters(), lr=lr_gnn)
    #gnn_loss = losses.TripletMarginLoss(margin=alpha)
    gnn_loss = nn.BCELoss()

   

    val_roc = []
    train_roc = []

    train_loss_gnn = []
  

    val_loss_gnn = []
   

    for epoch in tqdm(range(epochs), desc='Epochs', position=0):
        total_loss_gnn = 0
       

        y_true = []
        y_pred_score = []
        gnn.train()
        
        for sampled_data in train_loader:

            gnn_optim.zero_grad()
          

            # Move `sampled_data` to the respective `device`
            sampled_data = sampled_data.to(device)

            # Run `forward` pass of the model
            pred = gnn(sampled_data.x, sampled_data.edge_index,sampled_data.edge_label_index)
            
            # Get the ground truth labels from `sampled_data`
            ground_truth = sampled_data.edge_label.to(device)


            # Get lgnn loss
            loss = gnn_loss(pred.squeeze(),
                            ground_truth.to(torch.float))
            
            loss.backward()

            gnn_optim.step()

            pred_binary = torch.detach(pred).cpu().numpy()
            ground_truth_cpu = ground_truth.cpu().numpy()

            y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
            y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            total_loss_gnn += loss.item()
            

        train_roc.append(roc_auc_score(y_true, y_pred_score))
        train_loss_gnn.append(total_loss_gnn/len(train_loader))
       

        gnn.eval()
        
        with torch.no_grad():
            total_loss_gnn = 0
           

            y_pred_score = []
            y_true = []
            for sampled_data in val_loader:

                # Move `sampled_data` to the respective `device`
                sampled_data = sampled_data.to(device)

                # Run `forward` pass of the model
                pred = gnn(sampled_data.x, sampled_data.edge_index,sampled_data.edge_label_index)

                ground_truth = sampled_data.edge_label.to(device)

                total_loss_gnn += gnn_loss(pred.to(torch.float).squeeze(),
                                           ground_truth.to(torch.float)).item()
              
                pred_binary = torch.detach(pred).cpu().numpy()
                ground_truth_cpu = ground_truth.cpu().numpy()
                y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
                y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            val_roc.append(roc_auc_score(y_true, y_pred_score))
            val_loss_gnn.append(total_loss_gnn/len(val_loader))
       

            _, _, _, _, roc, ap = Testing_model_gnn(test_loader, gnn)
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
 

    # Adding legend
    plt.legend()
    datetime.datetime.now()
    # Saving the figure

    fig_name = f'/TrainVal_plot-{dataset_name}_batch_size{batch_size},alpha_{alpha}_lr:{lr_gnn}-{lr_lc}({ datetime.datetime.now().strftime("%m%d-%H:%M")}).png'
    fig_path = folder + fig_name
    plt.savefig(fig_path)
    acc, pre, recall, f1, roc, ap = Testing_model_gnn(test_loader, gnn)
    return acc, pre, recall, f1, roc, ap


def Training_Models_gcn_ML(train_loader, test_loader, val_loader, GNN_input, batch_size=0, dataset_name='', epochs=60, lr_gnn=0.0001, lr_lc=0.001, display=5, patience=5, alpha=0.1, folder='./results/figs'):
    # model definition

    best_roc = 0

    epochs_no_improve = 0

    GNN_hiddens = 256
    MLP_input = GNN_hiddens//2

    gnn = GNN1(input=GNN_input, hidden_channels=GNN_hiddens)
    gnn = gnn.to(device)

    gnn_optim = torch.optim.Adam(gnn.parameters(), lr=lr_gnn)
    gnn_loss = losses.TripletMarginLoss(margin=alpha)
    #gnn_loss = F.cross_entropy

   

    val_roc = []
    train_roc = []

    train_loss_gnn = []
  

    val_loss_gnn = []
   

    for epoch in tqdm(range(epochs), desc='Epochs', position=0):
        total_loss_gnn = 0
        total_loss_lc = 0

        y_true = []
        y_pred_score = []
        gnn.train()
        
        for sampled_data in train_loader:

            gnn_optim.zero_grad()
          

            # Move `sampled_data` to the respective `device`
            sampled_data = sampled_data.to(device)

            # Run `forward` pass of the model
            pred = gnn(sampled_data.x, sampled_data.edge_index,sampled_data.edge_label_index)
            
            # Get the ground truth labels from `sampled_data`
            ground_truth = sampled_data.edge_label.to(device)


            # Get lgnn loss
            loss = gnn_loss(pred,
                            ground_truth.to(torch.long))
            
            loss.backward()

           

          
            gnn_optim.step()

            pred_binary = torch.detach(pred).cpu().numpy()
            ground_truth_cpu = ground_truth.cpu().numpy()

            y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
            y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            total_loss_gnn += loss.item()
            

        train_roc.append(roc_auc_score(y_true, y_pred_score))
        train_loss_gnn.append(total_loss_gnn/len(train_loader))
       

        gnn.eval()
        
        with torch.no_grad():
            total_loss_gnn = 0
           

            y_pred_score = []
            y_true = []
            for sampled_data in val_loader:

                # Move `sampled_data` to the respective `device`
                sampled_data = sampled_data.to(device)

                # Run `forward` pass of the model
                pred = gnn(sampled_data.x, sampled_data.edge_index,sampled_data.edge_label_index)

                ground_truth = sampled_data.edge_label.to(device)

                total_loss_gnn += gnn_loss(pred.to(torch.float),
                                           ground_truth.to(torch.long)).item()
              
                pred_binary = torch.detach(pred).cpu().numpy()
                ground_truth_cpu = ground_truth.cpu().numpy()
                y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
                y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            val_roc.append(roc_auc_score(y_true, y_pred_score))
            val_loss_gnn.append(total_loss_gnn/len(val_loader))
       

            _, _, _, _, roc, ap = Testing_model_gnn(test_loader, gnn)
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
 

    # Adding legend
    plt.legend()
    datetime.datetime.now()
    # Saving the figure

    fig_name = f'/TrainVal_plot-{dataset_name}_batch_size{batch_size},alpha_{alpha}_lr:{lr_gnn}-{lr_lc}({ datetime.datetime.now().strftime("%m%d-%H:%M")}).png'
    fig_path = folder + fig_name
    plt.savefig(fig_path)
    acc, pre, recall, f1, roc, ap = Testing_model_gnn(test_loader, gnn)
    return acc, pre, recall, f1, roc, ap

def Training_Models_lc(train_loader, test_loader, val_loader, GNN_input, batch_size=0, dataset_name='', epochs=60, lr_gnn=0.0001, lr_lc=0.001, display=5, patience=5, alpha=0.1, folder='./results/figs'):
    # model definition

    best_roc = 0

    epochs_no_improve = 0


    lc = Classifier1(GNN_input*2, 128, 1)
    lc = lc.to(device)

    lc_optim = torch.optim.Adam(lc.parameters(), lr=lr_lc)
    
    lc_loss = nn.BCELoss()

    val_roc = []
    train_roc = []

    train_loss_lc = []


    val_loss_lc = []

    for epoch in tqdm(range(epochs), desc='Epochs', position=0):

        total_loss_lc = 0

        y_true = []
        y_pred_score = []
       
        lc.train()
        for sampled_data in train_loader:

           
            lc_optim.zero_grad()

            # Move `sampled_data` to the respective `device`
            sampled_data = sampled_data.to(device)
            # Get the ground truth labels from `sampled_data`
            ground_truth = sampled_data.edge_label.to(device)

            # Link classifier
            
            pred = lc(sampled_data.x,sampled_data.edge_label_index)
        
            loss2 = lc_loss(pred.squeeze(), ground_truth)
            loss2.backward()

            lc_optim.step()
            

            pred_binary = torch.detach(pred).cpu().numpy()
            ground_truth_cpu = ground_truth.cpu().numpy()

            y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
            y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            total_loss_lc += loss2.item()

        train_roc.append(roc_auc_score(y_true, y_pred_score))

        train_loss_lc.append(total_loss_lc/len(train_loader))


        lc.eval()
        with torch.no_grad():
            total_loss_gnn = 0
            total_loss_lc = 0

            y_pred_score = []
            y_true = []
            for sampled_data in val_loader:

                # Move `sampled_data` to the respective `device`
                sampled_data = sampled_data.to(device)

       

                # Get the ground truth labels from `sampled_data`
                ground_truth = sampled_data.edge_label.to(device)


                # Link classifier
                pred = lc(sampled_data.x,sampled_data.edge_label_index)
                total_loss_lc += lc_loss(pred.squeeze(), ground_truth).item()

                pred_binary = torch.detach(pred).cpu().numpy()
                ground_truth_cpu = ground_truth.cpu().numpy()
                y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
                y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            val_roc.append(roc_auc_score(y_true, y_pred_score))

            val_loss_lc.append(total_loss_lc/len(val_loader))

            _, _, _, _, roc, ap = Testing_model_lc(test_loader, lc)
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
    acc, pre, recall, f1, roc, ap = Testing_model_lc(test_loader, lc)
    return acc, pre, recall, f1, roc, ap

def Training_Models_lc_ML(train_loader, test_loader, val_loader, GNN_input, batch_size=0, dataset_name='', epochs=60, lr_gnn=0.0001, lr_lc=0.001, display=5, patience=5, alpha=0.1, folder='./results/figs'):
    # model definition

    best_roc = 0

    epochs_no_improve = 0


    lc = Classifier1(GNN_input*2, 128, 1)
    lc = lc.to(device)

    lc_optim = torch.optim.Adam(lc.parameters(), lr=lr_lc)
    lc_loss = losses.TripletMarginLoss(margin=alpha)
    

    val_roc = []
    train_roc = []

    train_loss_lc = []


    val_loss_lc = []

    for epoch in tqdm(range(epochs), desc='Epochs', position=0):

        total_loss_lc = 0

        y_true = []
        y_pred_score = []
       
        lc.train()
        for sampled_data in train_loader:

           
            lc_optim.zero_grad()

            # Move `sampled_data` to the respective `device`
            sampled_data = sampled_data.to(device)
            # Get the ground truth labels from `sampled_data`
            ground_truth = sampled_data.edge_label.to(device)

            # Link classifier
            
            pred = lc(sampled_data.x,sampled_data.edge_label_index)
        
            loss2 = lc_loss(pred.squeeze(), ground_truth)
            loss2.backward()

            lc_optim.step()
            

            pred_binary = torch.detach(pred).cpu().numpy()
            ground_truth_cpu = ground_truth.cpu().numpy()

            y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
            y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            total_loss_lc += loss2.item()

        train_roc.append(roc_auc_score(y_true, y_pred_score))

        train_loss_lc.append(total_loss_lc/len(train_loader))


        lc.eval()
        with torch.no_grad():
            total_loss_gnn = 0
            total_loss_lc = 0

            y_pred_score = []
            y_true = []
            for sampled_data in val_loader:

                # Move `sampled_data` to the respective `device`
                sampled_data = sampled_data.to(device)

       

                # Get the ground truth labels from `sampled_data`
                ground_truth = sampled_data.edge_label.to(device)


                # Link classifier
                pred = lc(sampled_data.x,sampled_data.edge_label_index)
                total_loss_lc += lc_loss(pred.squeeze(), ground_truth).item()

                pred_binary = torch.detach(pred).cpu().numpy()
                ground_truth_cpu = ground_truth.cpu().numpy()
                y_pred_score = np.hstack([y_pred_score, pred_binary.squeeze()])
                y_true = np.hstack([y_true, ground_truth_cpu.squeeze()])

            val_roc.append(roc_auc_score(y_true, y_pred_score))

            val_loss_lc.append(total_loss_lc/len(val_loader))

            _, _, _, _, roc, ap = Testing_model_lc(test_loader, lc)
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
    acc, pre, recall, f1, roc, ap = Testing_model_lc(test_loader, lc)
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

def Testing_model_gnn(test_loader, gnn):
    y_pred = []
    y_pred_score = []
    y_true = []
    gnn.eval()

    for sampled_data in test_loader:

        # Move `sampled_data` to the respective `device`
        sampled_data = sampled_data.to(device)

        # Run `forward` pass of the model
        pred = gnn(sampled_data.x, sampled_data.edge_index,sampled_data.edge_label_index)

        ground_truth = sampled_data.edge_label.to(device)

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

def Testing_model_lc(test_loader, lc):
    y_pred = []
    y_pred_score = []
    y_true = []
  
    lc.eval()
    for sampled_data in test_loader:

        # Move `sampled_data` to the respective `device`
        sampled_data = sampled_data.to(device)



        # Get the ground truth labels from `sampled_data`
        ground_truth = sampled_data.edge_label.to(device)

        # Link classifier
        pred = lc(sampled_data.x,sampled_data.edge_label_index)

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


# yst upg

def Exp_final(data, dataset_name, batch_size=8, lr_gnn=0.0001, lr_lc=0.001, alpha=0.5):
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
    epochs=100
    iters=3
    # Create a folder to save results
    folder_name = f"./results/testing/gnn1/new-{dataset_name}_{datetime.datetime.now().strftime('%m_%d(%H:%M)')}"
    os.mkdir(folder_name)

    # Log metadata to a file
    with open(folder_name+'/resultsMeta.txt', 'a') as f:
        f.write(f"\n***{datetime.datetime.now().strftime('%Y/%m/%d-(%H:%M)')}****\n")
    
    # List of batch sizes to iterate through
    batch_sizes = [8]
    batch_size=8
    
       # Perform experiments for a certain number of iterations
    # for iter in range(iters):
    #     print(f"iter {iter+1}  --> ")
        
    #     # Load data loaders
    #     train_loader, val_loader, test_loader = Get_data_loaders(
    #         data, batch_size=batch_size)
        
    #     # Train models and evaluate
    #     acc, pres, recall, f1, ap, roc = Training_Models_gcn_lc(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
    #                                                     epochs=epochs, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=5, alpha=alpha, folder=folder_name)
        
    #     # Log results to file
    #     r = f"datasetname={dataset_name}, GCN LC Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

    #     with open(folder_name+'/resultsMeta.txt', 'a') as f:
    #         f.writelines(r)
    #     print(r)
    

    # for iter in range(iters):
    #     print(f"iter {iter+1}  --> ")
        
    #     # Load data loaders
    #     train_loader, val_loader, test_loader = Get_data_loaders(
    #         data, batch_size=batch_size)
        
    #     # Train models and evaluate
    #     acc, pres, recall, f1, ap, roc = Training_Models_gcn(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
    #                                                     epochs=epochs, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=15, alpha=alpha, folder=folder_name)
        
    #     # Log results to file
    #     r = f"datasetname={dataset_name},  GCN Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

    #     with open(folder_name+'/resultsMeta.txt', 'a') as f:
    #         f.writelines(r)
    #     print(r)
        
        # Perform experiments for a certain number of iterations
    # for iter in range(iters):
    #     print(f"iter {iter+1}  --> ")
        
    #     # Load data loaders
    #     train_loader, val_loader, test_loader = Get_data_loaders(
    #         data, batch_size=batch_size)
        
    #     # Train models and evaluate
    #     acc, pres, recall, f1, ap, roc = Training_Models_gcn_ML(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
    #                                                     epochs=epochs, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=15, alpha=alpha, folder=folder_name)
        
    #     # Log results to file
    #     r = f"datasetname={dataset_name}, GCN ML Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

    #     with open(folder_name+'/resultsMeta.txt', 'a') as f:
    #         f.writelines(r)
    #     print(r) 

    for iter in range(iters):
        print(f"iter {iter+1}  --> ")
        
        # Load data loaders
        train_loader, val_loader, test_loader = Get_data_loaders(
            data, batch_size=batch_size)
        
        # Train models and evaluate
        acc, pres, recall, f1, ap, roc = Training_Models_lc(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
                                                        epochs=epochs, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=15, alpha=alpha, folder=folder_name)
        
        # Log results to file
        r = f"datasetname={dataset_name}, LC Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

        with open(folder_name+'/resultsMeta.txt', 'a') as f:
            f.writelines(r)
        print(r) 
    # for iter in range(iters):
    #     print(f"iter {iter+1}  --> ")
        
    #     # Load data loaders
    #     train_loader, val_loader, test_loader = Get_data_loaders(
    #         data, batch_size=batch_size)
        
    #     # Train models and evaluate
    #     acc, pres, recall, f1, ap, roc = Training_Models_lc_ML(train_loader, test_loader, val_loader, data.num_features, batch_size=batch_size,
    #                                                     epochs=epochs, lr_gnn=lr_gnn, lr_lc=lr_lc, display=2, patience=15, alpha=alpha, folder=folder_name)
        
    #     # Log results to file
    #     r = f"datasetname={dataset_name}, LC ML Ap: {ap:.4f}, roc: {roc:.4f} acc: {acc:.4f}, pre: {pres:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n"

    #     with open(folder_name+'/resultsMeta.txt', 'a') as f:
    #         f.writelines(r)
    #     print(r) 
   
      
    


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    
    df_dataset = pd.read_csv("results/datasets.csv",index_col=0)
    # Iterate through the sorted DataFrame and print the "datasetname" column
    for index, row in df_dataset.iterrows():
            dataset_name=row["dataset"]
    # for dataset_name in datasets:
            print(dataset_name)
            if index in [4,5,6,7,8]:
                pickle_file_path =f"data/datasets/processed/{dataset_name}"
                with open(pickle_file_path, "rb") as file:
                        data = pickle.load( file)
            
                            
                Exp_final(data,dataset_name.split('.')[0]+"_final",batch_size=8,lr_gnn=0.0001,lr_lc=0.0001)
                
