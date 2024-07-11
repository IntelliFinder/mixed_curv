import torch
import torch.nn as nn
from utils.manifold_utils import clip_hyperbolic

class SphericalComponent(nn.Module):
    def __init__(self, num_nodes, dim):
        super(SphericalComponent, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_nodes, dim) * 0.01) if dim > 0 else None
        if self.embeddings is not None:
            self.embeddings.manifold = 'spherical'
            self.project_to_sphere()

    def forward(self):
        return self.embeddings

    def project_to_sphere(self):
        if self.embeddings is not None:
            with torch.no_grad():
                self.embeddings.div_(torch.clamp(self.embeddings.norm(dim=1, keepdim=True), min=1e-8))

class EuclideanComponent(nn.Module):
    def __init__(self, num_nodes, dim):
        super(EuclideanComponent, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_nodes, dim) * 0.01) if dim > 0 else None
        if self.embeddings is not None:
            self.embeddings.manifold = 'euclidean'

    def forward(self):
        return self.embeddings

class HyperbolicComponent(nn.Module):
    def __init__(self, num_nodes, dim, init_range=3):
        super(HyperbolicComponent, self).__init__()
        self.dim = dim + 1  # Add one dimension for time component
        self.embeddings = nn.Parameter(torch.zeros(num_nodes, self.dim))
        self.embeddings.data[:, 1:] = torch.rand(num_nodes, dim) * 2 * init_range - init_range
        self.embeddings.manifold = 'hyperbolic'
        self.project_to_hyperboloid()

    def forward(self):
        return self.embeddings

    def project_to_hyperboloid(self):
        with torch.no_grad():
            self.embeddings.data = clip_hyperbolic(self.embeddings.data, max_norm=0.9)
