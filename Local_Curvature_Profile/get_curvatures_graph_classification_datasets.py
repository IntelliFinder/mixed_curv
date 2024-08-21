from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset, MalNetTiny
from torch_geometric.utils import to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from tqdm import tqdm
import pathlib
import warnings
import json
warnings.filterwarnings('ignore')

### Graph classification ###
mutag = TUDataset(root="data", name="MUTAG")
enzymes = TUDataset(root="data", name="ENZYMES")
proteins = TUDataset(root="data", name="PROTEINS")
collab = TUDataset(root="data", name="COLLAB")
imdb = TUDataset(root="data", name="IMDB-BINARY")
reddit = TUDataset(root="data", name="REDDIT-BINARY")
peptides = LRGBDataset(root="data", name="Peptides-func")
pascalvoc = LRGBDataset(root="data", name="PascalVOC-SP")
malnet = MalNetTiny(root="data")
# Requires OGB library but can break other dependencies
# from ogb.graphproppred import PygGraphPropPredDataset
# molhiv = PygGraphPropPredDataset(root='data', name="ogbg-molhiv")
# ppa = PygGraphPropPredDataset(root='data', name="ogbg-ppa")

### Datasets ###
datasets = {
    'graph_cls' : {
        'mutag' : mutag,
        'enzymes' : enzymes,
        'proteins' : proteins,
        'imdb' : imdb,
        'reddit': reddit,
        'peptides': peptides,
        'pascalvoc': pascalvoc,
        'malnet': malnet,
        # 'molhiv': molhiv,
        # 'ppa': ppa,
    }
}

def _get_graph_statistics(G):
    orc = OllivierRicci(G, alpha=0)
    orc.compute_ricci_curvature()

    all_curvatures = []
    for i, j in orc.G.edges:
        all_curvatures.append(orc.G[i][j]['ricciCurvature']['rc_curvature'])

    return all_curvatures

data_statistics = {}
for key in datasets['graph_cls']:
    print(f'[INFO] Calculating curvatures for {key}')
    dataset = datasets['graph_cls'][key]
    all_avg_curvatures = []
    for i in tqdm(range(len(dataset))):
        G = to_networkx(dataset[i])
        avg_curvatures = _get_graph_statistics(G)
        all_avg_curvatures.append(avg_curvatures)

    data_statistics[key] = all_avg_curvatures


pathlib.Path("results").mkdir(parents=True, exist_ok=True)
with open("results/gc_curvatures.json", "w") as f:
    json.dump(data_statistics, f)


