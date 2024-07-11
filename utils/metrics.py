import torch
from utils.manifold_utils import scalar_product_hyperbolic

def hyperbolic_distance(x, y):
    eps = 1e-7
    xy_inner = -x[:, 0] * y[:, 0] + torch.sum(x[:, 1:] * y[:, 1:], dim=1)
    cosh_distance = torch.clamp(-xy_inner, min=1 + eps)
    distance = torch.log(cosh_distance + torch.sqrt(torch.clamp(cosh_distance**2 - 1, min=eps)))
    return torch.clamp(distance, min=0, max=10)

def product_distance(x, y):
    x_s, x_e, x_h = x
    y_s, y_e, y_h = y
    num_nodes = x_h.shape[0]
    d_s = torch.zeros(num_nodes, device=x_h.device)
    d_e = torch.zeros(num_nodes, device=x_h.device)
    d_h = torch.zeros(num_nodes, device=x_h.device)

    if