import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import copy as cp

import torch
from torch.utils.data import random_split
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from PHEME_Dataset import PHEME_Dataset
from WEIBO_Dataset import WEIBO_Dataset

from utils.data_loader import *
from utils.eval_helper import *

from torch.utils.tensorboard import SummaryWriter

import time # to mesure the execution time.
import numpy as np
import pandas as pd
# tensorboad writer.
writer = SummaryWriter()

"""

The Bi-GCN is adopted from the original implementation from the paper authors 

Paper: Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
Link: https://arxiv.org/pdf/2001.06362.pdf
Source Code: https://github.com/TianBian95/BiGCN

"""
separator = "-" * 50
print(separator)
class TDrumorGCN(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(TDrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)
		rootindex = data.root_index
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
		batch_size = max(data.batch) + 1

		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)
		x = scatter_mean(x, data.batch, dim=0)

		return x


class BUrumorGCN(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(BUrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.BU_edge_index
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)

		rootindex = data.root_index
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
		batch_size = max(data.batch) + 1
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = scatter_mean(x, data.batch, dim=0)
		return x


class Net(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(Net, self).__init__()
		self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
		self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
		self.fc = torch.nn.Linear((out_feats+hid_feats) * 2, 2)

	def forward(self, data):
		TD_x = self.TDrumorGCN(data)
		BU_x = self.BUrumorGCN(data)
		x = torch.cat((TD_x, BU_x), 1)
		x = self.fc(x)
		x = F.log_softmax(x, dim=1)
		return x


def compute_test(loader, verbose=False, is_test=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	with torch.no_grad():
		for data in loader:
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)
			if args.multi_gpu:
				y = torch.cat([d.y for d in data]).to(out.device)
			else:
				y = data.y
			if verbose:
				print(F.softmax(out, dim=1).cpu().numpy())
			out_log.append([F.softmax(out, dim=1), y])
			loss_test += F.nll_loss(out, y).item()
	if is_test == True:
		return eval_deep(out_log, loader, is_test=True), loss_test
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--TDdroprate', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--BUdroprate', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=45, help='maximum number of epochs')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')

args = parser.parse_args()
print(args)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
	args.device = 'cuda'
if args.dataset == 'politifact' or args.dataset == 'gossipcop':
	dataset = FNNDataset(root='data/UPFD', feature=args.feature, empty=False, name=args.dataset,
					 transform=DropEdge(args.TDdroprate, args.BUdroprate))
	
if args.dataset == 'weibo' or args.dataset == 'WEIBO':
	dataset = WEIBO_Dataset()
if args.dataset == "pheme": 
	dataset = PHEME_Dataset()

args.num_classes = 2#dataset.num_classes
args.num_features = 768#dataset.num_features

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Net(args.num_features, args.nhid, args.nhid)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)

if not args.multi_gpu:
	BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
	BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
	base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
	optimizer = torch.optim.Adam([
		{'params': base_params},
		{'params': model.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
		{'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
	], lr=args.lr, weight_decay=args.weight_decay)
else:
	BU_params = list(map(id, model.module.BUrumorGCN.conv1.parameters()))
	BU_params += list(map(id, model.module.BUrumorGCN.conv2.parameters()))
	base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
	optimizer = torch.optim.Adam([
		{'params': base_params},
		{'params': model.module.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
		{'params': model.module.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
	], lr=args.lr, weight_decay=args.weight_decay)


# to store metrics and calculate mean + std

test_accs = []
test_precs = []
test_recalls = []
test_f1s = []
# new addition.
test_conf_matrices = []
test_auc_roc = []
test_classification_reports = []

# Record the start time
times = []
if __name__ == "__main__":

    for iter in range(args.iterations):
        # record the start time here.
        start_time = time.time()

        model.train()
        for epoch in range(args.epochs):
            out_log = []
            loss_train = 0.0
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                if not args.multi_gpu:
                    data = data.to(args.device)
                out = model(data)
                if args.multi_gpu:
                    y = torch.cat([d.y for d in data]).to(out.device)
                else:
                    y = data.y
                loss = F.nll_loss(out, y)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                out_log.append([F.softmax(out, dim=1), y])
            acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
            [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
            """ print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
    				  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
    				  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
    				  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}') """
    			
            writer.add_scalar(f"BIGCN-Loss-Train-{args.dataset}", loss_train, epoch)
            writer.add_scalar(f"BIGCN-Acc-Train-{args.dataset}", acc_train, epoch)
    
    			
            writer.add_scalar(f"BIGCN-Loss-Val-{args.dataset}", loss_val, epoch)
            writer.add_scalar(f"BIGCN-Acc-Val-{args.dataset}", acc_val, epoch)
    			
    
        [acc, f1_macro, f1_micro, precision, recall, auc, ap, cr, cm], test_loss = compute_test(test_loader, verbose=False, is_test=True)
        print(f'BIGCN-{args.dataset}-Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
    			  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
        print("classification report:", cr)
    
            
        #append and calcualte all metrics.
        test_conf_matrices.append(cm)
        test_accs.append(acc)
        test_f1s.append(f1_macro)
        test_precs.append(precision)
        test_recalls.append(recall)
        test_auc_roc.append(auc)
        test_classification_reports.append(cr)
        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        times.append(elapsed_time)

    
    final_test_acc = round(np.mean(test_accs), 5)
    std_test_acc = round(np.std(test_accs), 5)

    final_test_prec = round(np.mean(test_precs), 5)
    std_test_prec = round(np.std(test_precs), 5)

    final_test_recall = round(np.mean(test_recalls), 5)
    std_test_recall = round(np.std(test_recalls), 5)

    final_test_f1 = round(np.mean(test_f1s), 5)
    std_test_f1 = round(np.std(test_f1s), 5)

    final_test_auc_roc = round(np.mean(test_auc_roc), 5)
    std_test_auc_roc = round(np.std(test_auc_roc), 5)

    final_exec_time = round(np.mean(times), 5)
    
        
    final_classification_report = {
    'precision_0': round(np.mean([report['0']['precision'] for report in test_classification_reports]), 5),
    'recall_0': round(np.mean([report['0']['recall'] for report in test_classification_reports]), 5),
    'f1-score_0': round(np.mean([report['0']['f1-score'] for report in test_classification_reports]), 5),
    'precision_1': round(np.mean([report['1']['precision'] for report in test_classification_reports]), 5),
    'recall_1': round(np.mean([report['1']['recall'] for report in test_classification_reports]), 5),
    'f1-score_1': round(np.mean([report['1']['f1-score'] for report in test_classification_reports]), 5)
        }
    # Add mean confusion matrix to DataFrame
    final_confusion_matrix = np.mean(test_conf_matrices, axis=0)
    cm = final_confusion_matrix

    report = final_classification_report
    #confusion_matrix_columns = [f'CM_{i}' for i in range(final_confusion_matrix.shape[0])]

    # File path for the CSV file
    csv_file_path = 'BIGCN.csv'


    data = {
        'dataset': [args.dataset],
        'batch_size': [args.batch_size],
        'epochs': [args.epochs],
        
        'accuracy': [final_test_acc],
        'std_acc': [std_test_acc],

        'precesion': [final_test_prec],
        'std_prec': [std_test_prec],
        
        'recall': [final_test_recall],
        'std_recall': [std_test_recall],
        
        'f1_score': [final_test_f1],
        'std_f1': [std_test_f1],
        
        'auc_roc': [final_test_auc_roc],
        'std_auc_roc': [std_test_auc_roc],

        'exec_time': [final_exec_time],
        
        'tp': [cm[0, 0]],
        'tn': [cm[1, 1]],
        'fn': [cm[1, 0]],
        'fp': [cm[0, 1]],
        
        'precision_0': [report['precision_0']],
        'recall_0': [report['recall_0']],
        'f1_score_0': [report['f1-score_0']],
        'precision_1': [report['precision_1']],
        'recall_1': [report['recall_1']],
        'f1_score_1': [report['f1-score_1']],

    }
    df = pd.DataFrame(data)
    if os.path.exists(csv_file_path):
        # Load existing data
        existing_df = pd.read_csv(csv_file_path)

        # Append the new results to the existing data
        updated_df = pd.concat([existing_df, df], ignore_index=True)

        # Write the updated DataFrame to the CSV file
        updated_df.to_csv(csv_file_path, index=False)

    else:
        # Write the DataFrame to the CSV file
        df.to_csv(csv_file_path, index=False)

print(separator)
