from attrdict import AttrDict
from torch_geometric.datasets import WebKB, Planetoid, Amazon, Actor, HeterophilousGraphDataset, WikipediaNetwork
from torch_geometric.utils import to_undirected, dropout_edge
from experiments.node_classification import Experiment
from torch_geometric.data import Data

import time
import torch
import sys
import os
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf
from typing import Optional, Any

import torch_geometric.transforms as T
from custom_encodings import LocalCurvatureProfile

def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    # Taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "GCN",
    "display": True,
    "num_trials": 10,
    "eval_every": 1,
    "rewiring": "fosr",
    "num_iterations": 0, # By default do not apply rewiring
    "num_relations": 2,
    "patience": 100,
    "dataset": None,
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2,
    "sdrf_remove_edges" : False,
    "encoding": None,
    "mixed_curved_emb_path": "",
})


results = []
args = default_args
args += get_args_from_input()


if args.dataset is None:
    print("Provide the name of the dataset for NC task")
    sys.exit(0)

# Load a given dataset
if args.dataset == "texas":
    dataset = WebKB(root="data", name="Texas")
elif args.dataset == "wisconsin":
    dataset = WebKB(root="data", name="Wisconsin")
elif args.dataset == "cornell":
    dataset = WebKB(root="data", name="Cornell")
elif args.dataset == "cora":
    dataset = Planetoid(root="data", name="cora")
elif args.dataset == "citeseer":
    dataset = Planetoid(root="data", name="citeseer")
elif args.dataset == "pubmed":
    dataset = Planetoid(root="data", name="pubmed")
elif args.dataset == "amazon-photo":
    dataset = Amazon(root="data", name="Photo")
elif args.dataset == "amazon-computers":
    dataset = Amazon(root="data", name="Computers")
elif args.dataset == "actor":
    dataset = Actor(root="data")
elif args.dataset == "roman-empire":
    dataset = HeterophilousGraphDataset(root="data", name="Roman-empire")
elif args.dataset == "amazon-ratings":
    dataset = HeterophilousGraphDataset(root="data", name="Amazon-ratings")
elif args.dataset == "minesweeper":
    dataset = HeterophilousGraphDataset(root="data", name="Minesweeper")
elif args.dataset == "tolokers":
    dataset = HeterophilousGraphDataset(root="data", name="Tolokers")
elif args.dataset == "questions":
    dataset = HeterophilousGraphDataset(root="data", name="Questions")
elif args.dataset == "chameleon":
    dataset = WikipediaNetwork(root="data", name="chameleon")
elif args.dataset == "squirrel":
    dataset = WikipediaNetwork(root="data", name="squirrel")
else:
    print('Wrong dataset name')
    sys.exit(0)

# encode the dataset using the given encoding, if args.encoding is not None
if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB", "EGO", "EMB"]:
    if args.encoding == "LAPE":
        eigvecs = 8
        transform = T.AddLaplacianEigenvectorPE(k=eigvecs)
        print(f"Encoding Laplacian Eigenvector PE (k={eigvecs})")

    elif args.encoding == "RWPE":
        transform = T.AddRandomWalkPE(walk_length=16)
        print("Encoding Random Walk PE")

    elif args.encoding == "LDP":
        transform = T.LocalDegreeProfile()
        print("Encoding Local Degree Profile")

    elif args.encoding == "SUB":
        transform = T.RootedRWSubgraph(walk_length=10)
        print("Encoding Rooted RW Subgraph")

    elif args.encoding == "EGO":
        transform = T.RootedEgoNets(num_hops=2)
        print("Encoding Rooted Ego Nets")

    elif args.encoding == "LCP":
        lcp = LocalCurvatureProfile()
        transform = lcp.forward
        print(f"Encoding Local Curvature Profile (ORC)")

    elif args.encoding == "EMB":
        if not os.path.isfile(args.mixed_curved_emb_path):
            print("Provide a path to the embeddings")
            sys.exit(0)
        
        values = torch.load(args.mixed_curved_emb_path)

        def emb_transform(data):
            return add_node_attr(data, values)

        transform = emb_transform
        print('Encoding Mixed Curvature Embedding')


if args.encoding is not None:
    dataset.data = transform(dataset.data)

dataset.data.edge_index = to_undirected(dataset.data.edge_index)
datasets = {args.dataset: dataset}


def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


for key in datasets:
    accuracies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]

    start = time.time()
    if args.rewiring == "fosr":
        edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
        print(dataset.data.num_edges)
        print(len(dataset.data.edge_type))
    elif args.rewiring == "sdrf_bfc":
        curvature_type = "bfc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=args.sdrf_remove_edges, 
                is_undirected=True, curvature=curvature_type)
    elif args.rewiring == "borf":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf3(dataset.data, 
                loops=args.num_iterations, 
                remove_edges=False, 
                is_undirected=True,
                batch_add=args.borf_batch_add,
                batch_remove=args.borf_batch_remove,
                dataset_name=key,
                graph_index=0)
    elif args.rewiring == "barf_3":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf4(dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=key,
                    graph_index=i)
    elif args.rewiring == "barf_4":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf5(dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=key,
                    graph_index=i)
    elif args.rewiring == "sdrf_orc":
        curvature_type = "orc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, 
                is_undirected=True, curvature=curvature_type)
        
    elif args.rewiring == "dropedge":
        p = 0.8
        print(f"[INFO] Dropping edges with probability {p}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = dropout_edge(dataset[i].edge_index, dataset[i].edge_type, p=p, force_undirected=True)

    end = time.time()
    rewiring_duration = end - start
    print(f"Rewiring duration: {rewiring_duration}")    

    start = time.time()
    for trial in range(args.num_trials):
        print(f"TRIAL #{trial+1}")
        test_accs = []
        for i in range(args.num_splits):
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            test_accs.append(test_acc)
        test_acc = max(test_accs)
        accuracies.append(test_acc)
    end = time.time()
    run_duration = end - start

    log_to_file(f"RESULTS FOR {key} (filename: {args.mixed_curved_emb_path}, {args.layer_type}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "mixed_curved_emb_path": args.mixed_curved_emb_path,
        "euclidean_dim": args.euclidean_dim,
        "spherical_dim": args.spherical_dims,
        "hyperbolic_dim": args.hyperbolic_dims,
        "avg_accuracy": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "run_duration" : run_duration,
    })
    results_df = pd.DataFrame(results)
    with open(f'results/node_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)