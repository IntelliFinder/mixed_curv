from torch_geometric.datasets import (
    WebKB,
    Planetoid,
    Amazon,
    HeterophilousGraphDataset,
    WikipediaNetwork,
    Actor,
)
from torch_geometric.utils import to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import pathlib
import warnings
import json

warnings.filterwarnings("ignore")

### Node classification ###
cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")
amazon_photo = Amazon(root="data", name="Photo")
amazon_computers = Amazon(root="data", name="Computers")
actor = Actor(root="data")
roman_empire = HeterophilousGraphDataset(root="data", name="Roman-empire")
amazon_ratings = HeterophilousGraphDataset(root="data", name="Amazon-ratings")
minesweeper = HeterophilousGraphDataset(root="data", name="Minesweeper")
tolokers = HeterophilousGraphDataset(root="data", name="Tolokers")
questions = HeterophilousGraphDataset(root="data", name="Questions")
chameleon = WikipediaNetwork(root="data", name="chameleon")
squirrel = WikipediaNetwork(root="data", name="squirrel")

### Datasets ###
datasets = {
    "node_cls": {
        "cornell": cornell,
        "wisconsin": wisconsin,
        "texas": texas,
        "cora": cora,
        "citeseer": citeseer,
        "pubmed": pubmed,
        "amazon_photo": amazon_photo,
        "amazon_computers": amazon_computers,
        "actor": actor,
        "roman_empire": roman_empire,
        "amazon_ratings": amazon_ratings,
        "minesweeper": minesweeper,
        "tolokers": tolokers,
        "questions": questions,
        "chameleon": chameleon,
        "squirrel": squirrel,
    }
}


def _get_graph_statistics(G):
    orc = OllivierRicci(G, alpha=0)
    orc.compute_ricci_curvature()

    all_curvatures = []
    for i, j in orc.G.edges:
        all_curvatures.append(orc.G[i][j]["ricciCurvature"]["rc_curvature"])

    return all_curvatures


### Calculate statistics for node classification ###
data_statistics = {}
for key in datasets["node_cls"]:
    print(f"[INFO] Calculating curvatures for {key}")
    dataset = datasets["node_cls"][key]
    G = to_networkx(dataset.data)
    all_curvatures = _get_graph_statistics(G)
    data_statistics[key] = all_curvatures


pathlib.Path("results").mkdir(parents=True, exist_ok=True)
with open("results/nc_curvatures.json", "w") as f:
    json.dump(data_statistics, f)
