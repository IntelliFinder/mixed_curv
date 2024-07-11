import torch
import networkx as nx
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.product_manifold_embedding import ProductManifoldEmbedding
from optimizers.riemannian_sgd import RiemannianSGDProductManifoldOptimizer
from utils.loss_function import loss_function
from utils.train import train_product_manifold
from utils.metrics import scalar_product_hyperbolic
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, dropout_edge
import torch
import networkx as nx

def main():
    """Returns an hyperbolic embedding of the texas dataset graph"""
    texas = WebKB(root="data", name="Texas")
    #texas.data = to_undirected(texas.data)
    #texas.data = dropout_edge(texas.data, p=0.1)
    g = to_networkx(texas.data)
    
    D = nx.floyd_warshall_numpy(g.to_undirected())
    dist_matrix = torch.from_numpy(D).float()

    # Set up model and training parameters
    lr = 1e-1
    num_iter = 50000
    sphere_dim, euclidean_dim, hyperbolic_dim = 2, 2, 2
    model = ProductManifoldEmbedding(nx.number_of_nodes(g), sphere_dim, 
                                     euclidean_dim, hyperbolic_dim)

    optimizer = RiemannianSGDProductManifoldOptimizer(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-5)

    # Print initial state
    print("Initial state:")
    initial_embeddings = model()
    print("Spherical:", initial_embeddings[0])
    print("Euclidean:", initial_embeddings[1])
    print("Hyperbolic:", initial_embeddings[2])
    initial_loss = loss_function(initial_embeddings, dist_matrix)
    print(f"Initial loss: {initial_loss.item():.4f}")

    # Train model
    final_loss = train_product_manifold(model, dist_matrix, optimizer, scheduler, num_iterations=num_iter, early_stop_patience=500)

    # Print final state
    print("\nFinal state:")
    final_embeddings = model()
    print("Spherical:", final_embeddings[0])
    print("Euclidean:", final_embeddings[1])
    print("Hyperbolic:", final_embeddings[2])
    print(f"Final loss: {final_loss.item():.4f}")

    # Print norms of final embedded coordinates
    print_norms(final_embeddings, num_nodes)

    # Run tests
    run_tests(final_embeddings, final_loss, initial_loss)
    return final_embeddings

def print_norms(embeddings, num_nodes):
    print("\nNorms of final embedded coordinates:")
    spherical_norms = torch.norm(embeddings[0], dim=1) if embeddings[0] is not None else torch.tensor([])
    euclidean_norms = torch.norm(embeddings[1], dim=1) if embeddings[1] is not None else torch.tensor([])
    hyperbolic_norms = torch.norm(embeddings[2][:, 1:], dim=1) if embeddings[2] is not None else torch.tensor([])
    for i in range(num_nodes):
        print(f"Node {i+1}: "
              f"Spherical: {'N/A' if len(spherical_norms) == 0 else f'{spherical_norms[i].item():.6f}'}, "
              f"Euclidean: {'N/A' if len(euclidean_norms) == 0 else f'{euclidean_norms[i].item():.6f}'}, "
              f"Hyperbolic: {'N/A' if len(hyperbolic_norms) == 0 else f'{hyperbolic_norms[i].item():.6f}'}")

def run_tests(embeddings, final_loss, initial_loss):
    if torch.isfinite(final_loss) and torch.isfinite(initial_loss):
        if final_loss < initial_loss:
            print("\nTest passed: Final loss is lower than initial loss")
        else:
            print("\nOptimization did not improve the loss")
    else:
        print("\nOptimization resulted in invalid loss values")

    spherical_norms = torch.norm(embeddings[0], dim=1) if embeddings[0] is not None else torch.tensor([])
    hyperbolic_norms = torch.norm(embeddings[2][:, 1:], dim=1) if embeddings[2] is not None else torch.tensor([])

    if len(spherical_norms) > 0:
        if torch.all(torch.isfinite(spherical_norms)) and torch.all(torch.abs(spherical_norms - 1) < 1e-6):
            print("Test passed: All spherical embeddings are on the unit sphere")
        else:
            print("Warning: Not all spherical embeddings are on the unit sphere")

    if len(hyperbolic_norms) > 0:
        if torch.all(torch.isfinite(hyperbolic_norms)) and torch.all(hyperbolic_norms < 0.9):
            print("Test passed: All hyperbolic embeddings are finite and within the clipping boundary")
        else:
            print("Warning: Not all hyperbolic embeddings are finite or within the clipping boundary")

    # Verify hyperboloid constraint
    constraint_violation = torch.abs(scalar_product_hyperbolic(embeddings[2], embeddings[2]) + 1)
    print(f"\nMax constraint violation: {constraint_violation.max().item():.6f}")
    if torch.all(constraint_violation < 1e-5):
        print("Test passed: All embeddings satisfy the hyperboloid constraint")
    else:
        print("Warning: Not all embeddings satisfy the hyperboloid constraint")
if __name__ == "__main__":
    main()
