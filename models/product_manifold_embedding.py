import torch
import torch.nn as nn
from models.components import SphericalComponent, EuclideanComponent, HyperbolicComponent

class ProductManifoldEmbedding(nn.Module):
    def __init__(self, num_nodes, sphere_dim, euclidean_dim, hyperbolic_dim):
        super(ProductManifoldEmbedding, self).__init__()
        self.spherical = SphericalComponent(num_nodes, sphere_dim)
        self.euclidean = EuclideanComponent(num_nodes, euclidean_dim)
        self.hyperbolic = HyperbolicComponent(num_nodes, hyperbolic_dim)

    def forward(self):
        s = self.spherical() if self.spherical.embeddings is not None else None
        e = self.euclidean() if self.euclidean.embeddings is not None else None
        h = self.hyperbolic()
        return (s, e, h)
