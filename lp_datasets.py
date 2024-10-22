import os
import networkx as nx
import pickle
from gensim.models import Word2Vec
from node2vec import Node2Vec
import torch
from torch_geometric.data import Data


def lp_dataset(dataset_name):

        
        if os.path.exists('data/datasets/processed/'+dataset_name+'.pkl'):
            print("File exists!")
            with open('data/datasets/processed/'+dataset_name+'.pkl', "rb") as file:
                data = pickle.load( file)
                
    
        else:
            print("File do not exists! data preprocessing..")
            G = nx.read_pajek('data/datasets/raw/'+dataset_name+'_full.net')

            # Convert the NetworkX graph to PyTorch Geometric Data object
            data = Data()

            # Add empty features to the nodes
            node_embeddings = get_node_embaddings(G)
            data.x = torch.tensor(node_embeddings.vectors)  

            # Add edges to the graph
            edge_index = []
            for u, v,_ in G.edges:
                edge_index.append([int(u), int(v)])
                edge_index.append([int(v), int(u)]) # Assuming an undirected graph, add reverse edges as well
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data.edge_index = edge_index

            
            pickle_file_path ='data/datasets/processed/'+dataset_name+".pkl"  # Specify the path where you want to save the pickle file
            
            # Save the 'data' variable to a pickle file
            with open(pickle_file_path, "wb") as file:
                pickle.dump(data, file)
        return data

def get_node_embaddings(G,dimensions=64):
    # Precompute the random walks
    num_walks = 10
    walk_length = 80
    p = 1.0  # Return parameter
    q = 1.0  # In-out parameter

    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)

    # # Train the node2vec model
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # # Get the node embeddings
    node_embeddings = model.wv
    return(node_embeddings)


if __name__ == '__main__':

    # folder_path = "data/datasets/raw"  # Replace with the actual folder path
    # file_list = []

    # # Iterate over all files in the folder
    # for file_name in os.listdir(folder_path):
    #     # Check if the file name contains "_full"
    #     if "_full" in file_name:
    #         file_list.append(file_name)
    #         data = lp_dataset(file_name.split('_')[0])
    #         print(len(data.x))
    
    dataset_name = 'UAL'
    G=lp_dataset(dataset_name)
    print(G)
    # G = nx.read_pajek('data/datasets/raw/'+dataset_name+'_full.net')
    # for dimensions in [16,32,64,128,256]:
    # # Convert the NetworkX graph to PyTorch Geometric Data object
    #     data = Data()

    #     # Add empty features to the nodes
    #     node_embeddings = get_node_embaddings(G,dimensions=dimensions)
    #     data.x = torch.tensor(node_embeddings.vectors)  

    #     # Add edges to the graph
    #     edge_index = []
    #     for u, v,_ in G.edges:
    #         edge_index.append([int(u), int(v)])
    #         edge_index.append([int(v), int(u)]) # Assuming an undirected graph, add reverse edges as well
        
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     data.edge_index = edge_index

        
    #     pickle_file_path =f"data/datasets/processed/DataTest/{dataset_name}{dimensions}.pkl"  # Specify the path where you want to save the pickle file
    
    #     # Save the 'data' variable to a pickle file
    #     with open(pickle_file_path, "wb") as file:
    #         pickle.dump(data, file)    
